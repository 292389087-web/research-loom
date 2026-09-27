"""使用本机 Edge 无界面模式做真实页面回归，不下载浏览器。"""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ui-checks"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width":1440,"height":1100}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8010", wait_until="networkidle")
        page.get_by_role("heading", name="从读懂一篇论文， 到找到下一步研究。").wait_for()
        page.screenshot(path=str(OUT/"desktop.png"),full_page=True)
        page.locator('.starter[data-mode="plan"]').click()
        assert "研究切入点" in page.locator("#question").input_value()
        assert "active" in page.locator('.mode[data-mode="plan"]').get_attribute("class")
        page.locator(".profile-details summary").filter(has_text="我的研究背景").click()
        page.locator("#interests").fill("强化学习与多模态")
        if page.locator("[data-preview]").count():
            page.locator("[data-preview]").first.click()
            page.locator("#pageImage").wait_for(state="visible")
            expect(page.locator("#pageImage")).to_have_js_property("complete", True)
            assert page.locator("#pageImage").evaluate("image => image.naturalWidth > 0")
            page.screenshot(path=str(OUT/"source-viewer.png"),full_page=True)
            page.locator("#readThisPage").click()
            assert "active" in page.locator('.mode[data-mode="figure"]').get_attribute("class")
            assert page.locator("#figureControls").is_visible()
        page.set_viewport_size({"width":390,"height":844})
        page.screenshot(path=str(OUT/"mobile.png"),full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        browser.close()
    (OUT/"result.json").write_text(json.dumps({"page_errors":errors,"desktop":"1440x1100","mobile":"390x844"},ensure_ascii=False,indent=2),encoding="utf-8")
    assert not errors, errors
    print("Browser checks passed: starter, modes, profile, PDF viewer, page-to-task, mobile overflow, no JavaScript errors.")


if __name__ == "__main__":
    main()
