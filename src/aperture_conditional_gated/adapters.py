"""Native VLM integration; new gate states cannot be restored as legacy Full."""
from contextlib import contextmanager
import time
import torch
from aperture_rebuttal_cases.adapters import loaded as legacy_loaded, native_job
from aperture_rebuttal_cases.adapters import train_control as legacy_train
from .controller import ConditionalGatedController, FORMAT


@contextmanager
def loaded(unit,variant,restore=None):
    if variant['engine']!='conditional_gated':
        with legacy_loaded(unit,variant,restore) as triple:yield triple
        return
    from aperture_comprehensive.backend import loaded as native_loaded,decoder_sites
    from vigor_handoff.core import extract_basis
    from aperture_rebuttal_cases.data import load_roles
    from aperture_rebuttal_cases.generation import training_row
    job=native_job(unit,variant);controller=None
    with native_loaded(job) as (model,backend,_):
        try:
            if any(p.requires_grad for p in model.parameters()):raise ValueError('Gate base must be frozen')
            sites=[s for s in decoder_sites(model,job['layers']) if s[1] in job['sites']]
            if restore is not None:
                payload=torch.load(restore/'controller.pt',weights_only=True,map_location='cpu')
                if payload.get('format')!=FORMAT:raise ValueError('Not a CG-v1 checkpoint; no implicit legacy conversion')
                bases={(int(k.split(':')[0]),k.split(':')[1]):b for k,b in payload['bases'].items()}
                spectra={};seconds=0.
            else:
                support=load_roles(unit)['support']
                if unit['task_type']=='vqa':support=[training_row(r) for r in support]
                torch.cuda.synchronize();start=time.perf_counter()
                bases,spectra=extract_basis(model,sites,support,backend.loss_and_mask,
                                            job['rank'],job['basis_mode'],job['basis_seed'])
                torch.cuda.synchronize();seconds=time.perf_counter()-start
            controller=ConditionalGatedController(sites,bases,job['alpha'],job['gate_mode'],job['gate_eps'])
            if restore is not None:controller.restore(payload)
            controller.extraction_seconds=seconds;controller.basis_spectra=spectra
            model.eval();yield model,backend,controller
        finally:
            if controller is not None:controller.close()


def train_control(model,backend,controller,support,job):
    if job['engine']!='conditional_gated':return legacy_train(model,backend,controller,support,job)
    from vigor_handoff.core import fit_coefficients
    return fit_coefficients(model,controller,support,lambda r:backend.loss_and_mask(r,controller)[0],
                            steps=job['steps'],lr=job['lr'],l2=job['l2'],reduction=job['loss_reduction'])


def gate_evidence(backend,rows,controller):
    """Support/calibration-only distributions, not a score-based stopping rule."""
    if not isinstance(controller,ConditionalGatedController):return None
    totals={};controller.trace_enabled=True
    try:
        with torch.no_grad():
            for row in rows:
                controller.trace={};backend.candidate(row,row['label'],controller)
                for key,g in controller.trace.items():
                    if not g.numel() or not torch.isfinite(g).all():raise RuntimeError('Invalid native gate values')
                    x=totals.setdefault(key,{'n':0,'sum':0.,'sum2':0.,'min':1.,'max':0.,'below_05':0,'above_95':0})
                    x['n']+=g.numel();x['sum']+=float(g.sum());x['sum2']+=float(g.square().sum())
                    x['min']=min(x['min'],float(g.min()));x['max']=max(x['max'],float(g.max()))
                    x['below_05']+=int((g<.05).sum());x['above_95']+=int((g>.95).sum())
                controller.clear_mask()
    finally:controller.trace_enabled=False;controller.trace={};controller.clear_mask()
    for x in totals.values():
        x['mean']=x['sum']/x['n'];x['std']=(max(0.,x['sum2']/x['n']-x['mean']**2))**.5
    return {'scope':'selected visual tokens on supplied non-query rows','sites':totals}
