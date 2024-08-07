import cv
import pandas as pd
from click.testing import CliRunner
import pytest

class TestCV:
    def test_load_config(self):
        """Check configs are loaded correctly"""
        job_config = cv.load_config("cv/us.yml")
        assert "naive" in job_config
        assert job_config["region"] == "us"
        
    def test_run_cv(self, tmpdir):
        """Runs cv pipeline using a single set of parameters from cv/us.yml
           Run is stored in temporary directory using PyTest Fixture `tmpdir`
        """
        job_config = cv.load_config("cv/us.yml")
        cv.run_cv("naive", tmpdir, job_config)
        
    def test_filter_validation_days(self, tmp_path):
        """Tests split of validation days using tmp_path fixtures"""
        data_path = "covid19_spread/data/usa/data_cases.csv"
        output_path = tmp_path / "val.csv"
        cv.filter_validation_days(data_path, output_path, 7)
        
        original_df = pd.read_csv(data_path, index_col="region")
        filtered_df = pd.read_csv(output_path, index_col="region")
        assert (original_df.shape[1] - filtered_df.shape[1]) == 7
        
@pytest.mark.integration
class TestCVIntegration:
    def test_cv_naive_us(self, tmpdir):
        """Runs integration test with tmpdir fixture that's cleaned up after tests"""
        runner = CliRunner()
        result = runner.invoke(cv.cv, f"cv/us.yml naive -basedir {tmpdir}")
        assert result.exit_code == 0
        
    def test_cv_naive_basedate(self, tmpdir):
        """Runs integration test with tmpdir fixture that's cleaned up after tests"""
        runner = CliRunner()
        result = runner.invoke(
            cv.cv, f"cv/us.yml naive -basedir {tmpdir} -basedate 2020-04-01"
        )
        assert result.exit_code == 0
        
        
        
