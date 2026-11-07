"""CLI and native-loader contracts; no model weights or GPU results."""
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def fake_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.mark.parametrize('engine', ['lora', 'dora'])
@pytest.mark.parametrize('restore', [False, True])
def test_adapter_reuses_processor_from_backend_not_native_third_return(monkeypatch, tmp_path, engine, restore):
    """The real native loader returns (model, Backend, controller), NOT processor."""
    from aperture_rebuttal_cases import adapters
    processor = object()
    calls = []

    class Model:
        def named_parameters(self): return []
        def gradient_checkpointing_enable(self, **kwargs): pass
        def enable_input_require_grads(self): pass
        def eval(self): pass

    model = Model()

    @contextmanager
    def native_loaded(job):
        yield model, SimpleNamespace(processor=processor), None

    def backend(job, model, supplied_processor):
        calls.append(supplied_processor)
        return SimpleNamespace(processor=supplied_processor)

    fake_module(monkeypatch, 'aperture_comprehensive.backend', loaded=native_loaded,
                Backend=backend, decoder_sites=lambda *a: [], lora_targets=lambda m: ['decoder.q_proj'])
    fake_module(monkeypatch, 'vigor_handoff.core', ResidualController=object, extract_basis=lambda *a: None)
    fake_module(monkeypatch, 'peft', LoraConfig=lambda **kw: kw,
                get_peft_model=lambda model, config: model,
                PeftModel=SimpleNamespace(from_pretrained=lambda model, *a, **kw: model))
    variant = dict(engine=engine, family=engine, lora_rank=4, lora_alpha=8, lora_dropout=.05)
    with adapters.loaded({'family':'qwen2_5_vl'}, variant, tmp_path if restore else None) as (_, active_backend, _):
        assert active_backend.processor is processor
    assert calls == [processor]


@pytest.fixture
def cli_repo(tmp_path, monkeypatch):
    """Use the exact checked-in comprehensive config; omit private data/weights."""
    import json
    from aperture_rebuttal_cases import cli
    repo = tmp_path / 'repo'
    (repo / 'configs').mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    for name in ('aperture_comprehensive_v1.json', 'aperture_reviewer_cases_v1.json'):
        (repo/'configs'/name).write_bytes((source/'configs'/name).read_bytes())
    monkeypatch.setattr(cli, 'REPO', repo)
    return cli, repo


@pytest.mark.parametrize('case,units,fits', [
    ('E01',330,1650), ('E02',18,450), ('E03',36,468), ('E04',9,72),
    ('E05',18,90), ('E06',2,10), ('E07',6,24), ('E08',0,0), ('E09',15,0)])
def test_actual_case_plan_commands(cli_repo, capsys, case, units, fits):
    import json
    cli, _ = cli_repo
    assert cli.main(['plan','--case',case]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['units'] == units and data['fits'] == fits


def test_advertised_e02_single_unit_is_exact_and_dry(cli_repo, capsys, monkeypatch):
    import json
    cli, _ = cli_repo
    monkeypatch.setattr(cli, 'execute', lambda *a, **kw: pytest.fail('Dry run launched execution'))
    args = ['run','--case','E02','--model','q25','--domain','camelyon17-hospital_2','--seed','0','--gpu','0']
    assert cli.main(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['status'] == 'dry_run'
    assert data['units'] == ['q25--camelyon17-hospital_2--n16--s0']


def test_e03_requires_native_domain_and_unit_for_one_budget(cli_repo, capsys):
    import json
    cli, _ = cli_repo
    assert cli.main(['run','--case','E03','--model','q25','--domain','hospital_2','--seed','0']) == 0
    assert len(json.loads(capsys.readouterr().out)['units']) == 3
    assert cli.main(['run','--case','E03','--unit','q25--hospital_2--n32--s0']) == 0
    assert json.loads(capsys.readouterr().out)['units'] == ['q25--hospital_2--n32--s0']
    assert cli.main(['run','--case','E03','--domain','camelyon17-hospital_2']) == 2
    assert 'No registered' in capsys.readouterr().err


@pytest.mark.parametrize('case',['E03','E07'])
def test_preparation_without_execute_is_only_dry_run(cli_repo, capsys, case):
    import json
    cli, _ = cli_repo
    assert cli.main(['prepare','--case',case]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'dry_run'


def test_multi_gpu_flag_rejected_single_gpu_index_required(cli_repo, capsys):
    cli, _ = cli_repo
    assert cli.main(['run','--case','E02','--gpu','0,1']) == 2
    assert 'one nonnegative GPU index' in capsys.readouterr().err


def test_e08_needs_execute_to_read_and_analyze_evidence(cli_repo, capsys):
    import json
    cli, _ = cli_repo
    assert cli.main(['run','--case','E08']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'dry_run'


def test_incomplete_case_cannot_be_summarized_as_success(cli_repo, capsys):
    cli, _ = cli_repo
    assert cli.main(['summarize','--case','E02']) == 2
    assert 'Incomplete case' in capsys.readouterr().err
