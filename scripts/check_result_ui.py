"""对真实已完成结果进行只读交互检查；不把自动测试标成学术人工复核。"""
from pathlib import Path
import json
import httpx
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ui-checks"


def main():
    runs = httpx.get("http://127.0.0.1:8010/api/runs").json()
    run = next(r for r in runs if r["status"] == "completed" and r["kind"] == "research" and r["request"]["mode"] == "plan")
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width":1440,"height":1050},device_scale_factor=1)
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.goto("http://127.0.0.1:8010",wait_until="networkidle")
        page.locator(f'[data-run="{run["id"]}"]').click()
        page.locator(".direction").first.wait_for()
        assert page.locator("#traceLink").get_attribute("href")==f'/api/runs/{run["id"]}/trace'
        page.locator("#progressPanel summary").click()
        with page.expect_popup() as popup:
            page.locator("#traceLink").click()
        trace_page=popup.value
        trace_page.wait_for_load_state()
        assert trace_page.url.endswith(f'/api/runs/{run["id"]}/trace')
        trace_page.close()
        page.locator("#results").scroll_into_view_if_needed()
        page.screenshot(path=str(OUT/"research-plan.png"),full_page=True)
        page.locator('[data-tab="experiment"]').click()
        assert page.locator(".finding").count() > 0
        page.screenshot(path=str(OUT/"experiment-evidence.png"),full_page=True)
        page.locator(".citation").first.click()
        expect(page.locator("#pageImage")).to_have_js_property("complete", True)
        assert page.locator("#pageImage").evaluate("image => image.naturalWidth > 0")
        assert page.locator("#evidenceHighlight").is_visible()
        assert len(page.locator("#quoteText").inner_text()) > 20
        page.screenshot(path=str(OUT/"citation-highlight.png"),full_page=True)
        page.locator("#closeViewer").click()
        page.locator("[data-review]").first.click()
        assert page.locator("#reviewDialog").is_visible()
        page.get_by_role("button",name="取消",exact=True).click()
        with page.expect_download() as event:
            page.locator("#exportMd").click()
        download = event.value
        download.save_as(OUT/"checked-export.md")
        assert "研究记录" in (OUT/"checked-export.md").read_text(encoding="utf-8")
        page.set_viewport_size({"width":390,"height":844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth+1")
        browser.close()
    (OUT/"result-interactions.json").write_text(json.dumps({"run_id":run["id"],"errors":errors,"checks":["history restore","trace link popup","research directions","experiment tab","citation page highlight","review dialog","markdown download","mobile overflow"]},ensure_ascii=False,indent=2),encoding="utf-8")
    assert not errors, errors
    print("Result UI passed",run["id"])


if __name__ == "__main__":
    main()
