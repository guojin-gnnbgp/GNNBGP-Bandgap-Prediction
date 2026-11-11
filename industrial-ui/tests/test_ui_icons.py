"""Regression checks for the offline industrial SVG icon system (not model accuracy)."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "studio/static/index.html").read_text(encoding="utf-8")
JS = (ROOT / "studio/static/app.js").read_text(encoding="utf-8")
CSS = (ROOT / "studio/static/style.css").read_text(encoding="utf-8")


class TestUIIcons(unittest.TestCase):
    def test_sprite_and_references(self):
        defined = set(re.findall(r'<symbol id="i-([^"]+)"', HTML))
        used = set(re.findall(r'<use href="#i-([^"]+)"', HTML))
        self.assertGreaterEqual(len(defined), 30)
        self.assertEqual(defined & used, used)
        self.assertEqual(HTML.count("<svg"), HTML.count("</svg>"))
        self.assertIn(".icon-defs", CSS)
        self.assertIn(".ui-icon", CSS)

    def test_windows_style_multicolour_file_icons(self):
        self.assertIn('linearGradient id="bgp-folder"', HTML)
        self.assertIn('linearGradient id="bgp-paper"', HTML)
        self.assertIn('fill="url(#bgp-folder)"', HTML)
        self.assertIn('fill="url(#bgp-paper)"', HTML)
        for filename_icon in (
            "file-cif", "file-poscar", "file-vasp", "file-xyz",
            "file-json", "file-csv", "file-log", "file-pdf",
        ):
            with self.subTest(icon=filename_icon):
                self.assertIn(f'<symbol id="i-{filename_icon}"', HTML)
        self.assertIn('function fileIconFor(filename)', JS)
        self.assertIn('iconHTML(fileIconFor(m.filename))', JS)
        self.assertIn('iconHTML(fileIconFor(f.path)', JS)
        self.assertIn('.artifact-name', CSS)

    def test_toolbar_icons_in_correct_buttons(self):
        expected = {
            "tool-import": "file-up",
            "tool-example": "folder-open",
            "tool-run": "play",
            "tool-stop": "stop",
            "tool-retry": "rotate-cw",
            "tool-refresh": "refresh-cw",
            "tool-report": "report",
            "tool-env": "settings",
            "inspector-run": "play",
            "result-export": "download",
        }
        for button_id, icon_name in expected.items():
            with self.subTest(button_id=button_id):
                pattern = rf'<button[^>]*id="{button_id}"[^>]*>.*?#i-{icon_name}'
                self.assertRegex(HTML, pattern)

    def test_source_contract_untouched(self):
        self.assertIn("/mnt/d/program/gnnbgp", (ROOT / "scripts/start_wsl.sh").read_text())
        self.assertIn("bgp predict", (ROOT / "README.md").read_text())
        self.assertIn('function menuItems(which)', JS)
        self.assertIn("menuIconNames", JS)
        self.assertNotIn("https://fonts.googleapis.com", HTML)
        self.assertNotIn("https://cdn.", HTML)


if __name__ == "__main__":
    unittest.main()
