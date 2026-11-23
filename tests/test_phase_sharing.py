import pytest

from kiln.config import RunConfig, TierConfig
from kiln.engine.base import Request
from kiln.server.jobs import JobManager


@pytest.mark.parametrize("owner_phase,borrower_phase", [("prefill", "decode"), ("decode", "prefill")])
def test_shared_phase_uses_borrower_for_baseline_reload_and_repeats(tmp_path, owner_phase, borrower_phase):
    owner_tier = TierConfig(name=owner_phase, phase=owner_phase)
    borrower_tier = TierConfig(name=borrower_phase, phase=borrower_phase, share_engine=owner_phase,
                               repeats=3, screen_repeats=1, screen_min_speedup=0.5)
    run = RunConfig(workdir=str(tmp_path), tiers=[owner_tier, borrower_tier], micro={"enabled": False})
    mgr = JobManager(run, [Request([1, 3, 5], 6)])
    owner = mgr.tiers[owner_phase]
    borrower = mgr.tiers[borrower_phase]
    owner._boot()
    borrower._boot()
    try:
        for phase, worker in [(borrower_phase, borrower), (owner_phase, owner)]:
            candidate = (
                "import os\n"
                f"assert os.environ['KILN_PD_ROLE'] == {phase!r}\n"
                "SPEEDUP = 2.0\nPREFILL_SPEEDUP = 3.0\n"
            )
            job = mgr.submit("test", phase, candidate)
            worker.execute_job(job)
            assert job.status == "done", job.error
            assert job.speedup == pytest.approx(3.0 if phase == "prefill" else 2.0, rel=0.05)
            assert (job.decode_tps == 0) == (phase == "prefill")
            if worker is borrower:
                assert job.extra["screening"]["total_repeats"] == 3
    finally:
        owner.engine.shutdown()
