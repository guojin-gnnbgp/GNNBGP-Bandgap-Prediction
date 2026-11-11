"""UI regression checks. Do not interpret mock-CLI tests as scientific verification."""
from pathlib import Path
import unittest
import re
ROOT=Path(__file__).resolve().parents[1]
HTML=(ROOT/"studio/static/index.html").read_text(encoding="utf-8")
JS=(ROOT/"studio/static/app.js").read_text(encoding="utf-8")
PY=(ROOT/"studio/server.py").read_text(encoding="utf-8")

class TestFeatures(unittest.TestCase):
 def test_business_categories_match_engine(self):
  for label in ("机器学习带隙误差修正","GPU 带隙计算","数据驱动泛函带隙计算"):
   self.assertIn(label,HTML)
  self.assertIn('["gnn-hse06","GNN-HSE06',JS)
  self.assertIn('["abacus-hse06","ABACUS-HSE06',JS)
  self.assertIn('["abacus-pbe0","ABACUS-PBE0',JS)
  self.assertIn('["cider-pbe0","CIDER-PBE0',JS)
  self.assertIn('id="compute-category"',HTML)
  self.assertIn('function updateCalculationCategory()',JS)
 def test_tree_delete_with_history_protection(self):
  self.assertIn('class="tree-delete"',JS)
  self.assertIn('data-remove-material=',JS)
  self.assertIn('async function removeMaterial(mid)',JS)
  self.assertIn('ALTER TABLE materials ADD COLUMN archived_at TEXT',PY)
  self.assertIn('WHERE archived_at IS NULL',PY)
  self.assertIn('allow_archived=True',PY)
 def test_report_is_inline_not_new_page(self):
  self.assertIn('data-view="report"',HTML)
  self.assertIn('id="view-report"',HTML)
  self.assertIn('function renderReport()',JS)
  self.assertIn('activateView("report")',JS)
  self.assertNotIn('window.open(',JS)
  self.assertIn('id="report-save"',HTML)
 def test_existing_static_contract(self):
  self.assertIn('data-view="structure"',HTML)
  self.assertIn('id="jobs-body"',HTML)
  self.assertIn('id="crystal-canvas"',HTML)
  self.assertIn('bgp predict', (ROOT/"README.md").read_text(encoding="utf-8"))
if __name__=="__main__":
 unittest.main()
