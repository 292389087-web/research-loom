"""真实 Edge 页面检查；不发模型请求、不登记虚构科研实验。"""

import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

out = Path(__file__).resolve().parents[1] / "data" / "ui-checks"
out.mkdir(parents=True, exist_ok=True)
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1050})
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:8010/workspace", wait_until="networkidle")
    page.get_by_role("heading", name="我的参考文献").wait_for()
    page.screenshot(path=str(out / "workspace-library.png"), full_page=True)
    page.locator('[data-tab="citations"]').click()
    options = page.locator("#draftList option").all()
    if len(options) > 1:
        page.locator("#draftList").select_option(index=1)
        expect(page.locator("#draftText")).to_have_value(re.compile(".+"))
        page.locator("#auditInput").fill(
            json.dumps(
                {
                    "operation": "sum",
                    "parts": [
                        {
                            "name": "a",
                            "value": "0.1",
                            "unit": "unit",
                            "source": "engineering fixture",
                        },
                        {
                            "name": "b",
                            "value": "0.2",
                            "unit": "unit",
                            "source": "engineering fixture",
                        },
                    ],
                    "total": {
                        "name": "total",
                        "value": "0.3",
                        "unit": "unit",
                        "source": "engineering fixture",
                    },
                    "tolerance": "0",
                }
            )
        )
        page.locator("#audit").click()
        expect(page.locator("#auditResult")).to_contain_text('"passed": true')
        assert json.loads(page.locator("#auditResult").inner_text())["passed"]
    for tab in ("citations", "lab", "bi"):
        page.locator(f'[data-tab="{tab}"]').click()
        assert page.locator("#" + tab).is_visible()
        page.screenshot(path=str(out / f"workspace-{tab}.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    for tab in ("library", "citations", "lab", "bi"):
        page.locator(f'[data-tab="{tab}"]').click()
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth+1"
        ), tab
    page.screenshot(path=str(out / "workspace-mobile.png"), full_page=True)
    browser.close()
(out / "workspace-result.json").write_text(
    json.dumps(
        {"errors": errors, "tabs": 4, "viewports": ["1440x1050", "390x844"]}, indent=2
    ),
    encoding="utf-8",
)
assert not errors, errors
print("Workspace browser checks passed: four tabs, two viewports, no JS errors.")
