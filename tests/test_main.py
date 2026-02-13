"""
Testes funcionais para ClassificationAnalyzer.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ClassificationAnalyzer, main


class TestClassificationAnalyzer:
    """Testes para a classe ClassificationAnalyzer."""

    def test_init_defaults(self):
        analyzer = ClassificationAnalyzer()
        assert analyzer.data is None
        assert analyzer.model is None
        assert analyzer.results == {}

    def test_load_data_generates_synthetic(self):
        analyzer = ClassificationAnalyzer()
        df = analyzer.load_data()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1000, 4)
        assert list(df.columns) == ['feature1', 'feature2', 'feature3', 'target']
        assert set(df['target'].unique()).issubset({0, 1})

    def test_load_data_accepts_custom_dataframe(self):
        analyzer = ClassificationAnalyzer()
        custom = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [0.5, 1.5, 2.5, 3.5],
            'target': [0, 1, 0, 1],
        })
        result = analyzer.load_data(data=custom)
        assert result.shape == (4, 3)
        assert analyzer.data is result

    def test_load_data_deterministic(self):
        """Seed=42 deve produzir dados identicos entre chamadas."""
        a1 = ClassificationAnalyzer()
        a2 = ClassificationAnalyzer()
        d1 = a1.load_data()
        d2 = a2.load_data()
        pd.testing.assert_frame_equal(d1, d2)

    def test_analyze_returns_results(self):
        analyzer = ClassificationAnalyzer()
        analyzer.load_data()
        results = analyzer.analyze()
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'statistics' in results
        assert isinstance(results['accuracy'], float)
        assert 0.0 <= results['accuracy'] <= 1.0

    def test_analyze_accuracy_above_baseline(self):
        """Com dados sinteticos correlacionados, accuracy deve superar 60%."""
        analyzer = ClassificationAnalyzer()
        analyzer.load_data()
        results = analyzer.analyze()
        assert results['accuracy'] > 0.6

    def test_analyze_auto_loads_data(self):
        """analyze() deve carregar dados automaticamente se nao carregados."""
        analyzer = ClassificationAnalyzer()
        results = analyzer.analyze()
        assert analyzer.data is not None
        assert 'accuracy' in results

    def test_model_is_fitted(self):
        analyzer = ClassificationAnalyzer()
        analyzer.load_data()
        analyzer.analyze()
        assert analyzer.model is not None
        assert hasattr(analyzer.model, 'predict')
        assert hasattr(analyzer.model, 'feature_importances_')

    def test_feature_importances_sum_to_one(self):
        analyzer = ClassificationAnalyzer()
        analyzer.load_data()
        analyzer.analyze()
        total = sum(analyzer.model.feature_importances_)
        assert abs(total - 1.0) < 1e-6

    def test_visualize_saves_png(self):
        analyzer = ClassificationAnalyzer()
        analyzer.load_data()
        analyzer.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_output.png')
            analyzer.visualize(output_path=path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0

    def test_visualize_without_model(self):
        """visualize() sem analyze() deve funcionar (sem grafico de importancia)."""
        analyzer = ClassificationAnalyzer()
        analyzer.load_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_no_model.png')
            analyzer.visualize(output_path=path)
            assert os.path.isfile(path)

    def test_visualize_auto_loads_data(self):
        """visualize() deve carregar dados automaticamente se nao carregados."""
        analyzer = ClassificationAnalyzer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_auto.png')
            analyzer.visualize(output_path=path)
            assert os.path.isfile(path)
            assert analyzer.data is not None

    def test_classification_report_format(self):
        analyzer = ClassificationAnalyzer()
        analyzer.load_data()
        results = analyzer.analyze()
        report = results['classification_report']
        assert isinstance(report, str)
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1-score' in report

    def test_statistics_contains_describe(self):
        analyzer = ClassificationAnalyzer()
        analyzer.load_data()
        results = analyzer.analyze()
        stats = results['statistics']
        assert isinstance(stats, pd.DataFrame)
        assert 'feature1' in stats.columns
        assert 'mean' in stats.index


class TestMain:
    """Testes para a funcao main()."""

    def test_main_returns_analyzer(self):
        analyzer = main()
        assert isinstance(analyzer, ClassificationAnalyzer)
        assert analyzer.model is not None
        assert 'accuracy' in analyzer.results

    def test_main_model_has_predictions(self):
        analyzer = main()
        X_test = analyzer.data.drop('target', axis=1).iloc[:5]
        preds = analyzer.model.predict(X_test)
        assert len(preds) == 5
        assert all(p in [0, 1] for p in preds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
