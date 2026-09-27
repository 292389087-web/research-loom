"""Read-only browser regression for the human-reviewed experiment workspace."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

out = Path(__file__).resolve().parents[1] / 'data' / 'ui-checks' / 'experiment-assistant'
out.mkdir(parents=True, exist_ok=True)
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width':1440,'height':1000})
    page.on('pageerror', lambda err: errors.append(str(err)))
    page.goto('http://127.0.0.1:8010/workspace', wait_until='networkidle')
    page.get_by_role('button', name='03 实验助手').click()
    expect(page.get_by_role('heading', name='准备本轮实验')).to_be_visible()
    expect(page.locator('#maxTrials')).to_have_value('1')
    values = json.loads(page.locator('#parameters').input_value())
    assert all(len(v)==1 for v in values.values())
    page.locator('#studyList').select_option('study_aad8c37916d8496a')
    expect(page.locator('#studyDetail')).to_contain_text('10-day')
    page.locator('#refreshNotebook').click()
    expect(page.locator('#notebookResult')).to_contain_text('协议一致')
    page.screenshot(path=str(out/'desktop.png'), full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=str(out/'mobile.png'), full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1')
    browser.close()
(out/'result.json').write_text(json.dumps({'page_errors':errors,'passed':not errors},ensure_ascii=False,indent=2),encoding='utf8')
assert not errors, errors
print('Experiment assistant UI: labels, one-round form, historical notebook, mobile layout and JavaScript passed.')
