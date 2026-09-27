"""Isolated local Edge UI regression; synthetic records, no model or training calls."""
import argparse
import json
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from loom.store import Store, dump, now
from loom.experiments import StudySpec, create_study


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8011')
    parser.add_argument('--data-dir',type=Path,required=True)
    args=parser.parse_args()
    out=ROOT/'data/evals/batch-experiments-20260919/ui';out.mkdir(parents=True,exist_ok=True)
    store=Store(args.data_dir)
    source=args.data_dir/'fixture-source';source.mkdir(exist_ok=True)
    (source/'train.py').write_text('raise RuntimeError("UI fixture must never train")',encoding='utf-8')
    adapter={'id':'ui-design-fixture','name':'界面验证工程（合成记录）','root':str(source.resolve()),'entrypoint':'train.py','source_files':['train.py'],
        'parameters':{'reward':[1,2],'epochs':[2,4],'days':[10,26]},'parameter_groups':{'reward':'reward','epochs':'hyperparameter','days':'protocol'},
        'parameter_labels':{'reward':'迁移惩罚系数','epochs':'PPO 更新轮数','days':'训练天数'},'metrics':['energy','temperature'],'fixture':True}
    folder=args.data_dir/'trainers';folder.mkdir(exist_ok=True)
    (folder/'ui-design-fixture.json').write_text(dump(adapter),encoding='utf-8')
    b=create_study(store,StudySpec(name='UI 合成基准',trainer_id=adapter['id'],parameters={'reward':[1],'epochs':[2],'days':[10]},objective='energy',
        protocol='synthetic UI regression only',review_mode=True,max_trials=1,evaluation_metrics=['energy','temperature']))
    store.execute("UPDATE studies SET status='completed' WHERE id=?",(b['id'],))
    store.execute('INSERT INTO trials VALUES (?,?,?,?,?,?,?,?)',('trial_'+b['id'],b['id'],0,dump({'reward':1,'epochs':2,'days':10}),'completed',dump({'objective_mean':100,'metrics_mean':{'energy':100},'feasible':True,'repeat_count':1}),None,now()))
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1050})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(args.url+'/workspace',wait_until='networkidle')
        page.locator('[data-tab="lab"]').click()
        page.locator('#trainer').select_option(adapter['id'])
        page.locator('#studyName').fill('UI 批量联合实验')
        page.locator('#protocol').fill('synthetic fixed protocol; UI regression only')
        page.locator('#parameterGroups [data-param="reward"]').fill('1,2')
        page.locator('#parameterGroups [data-param="epochs"]').fill('2,4')
        page.locator('#previewBatch').click()
        expect(page.locator('#batchPreview')).to_contain_text('4 组配置')
        expect(page.locator('#batchPreview')).to_contain_text('12 次训练')
        expect(page.locator('#createStudy')).to_be_disabled()
        page.locator('#confirmBatch').check()
        expect(page.locator('#createStudy')).to_be_enabled()
        page.locator('#studyName').fill('UI 批量联合实验（已核对）')
        expect(page.locator('#createStudy')).to_be_disabled()
        page.locator('#previewBatch').click()
        expect(page.locator('#batchPreview')).to_contain_text('4 组配置')
        page.locator('#confirmBatch').check()
        page.locator('#createStudy').click()
        expect(page.locator('#studyDetail')).to_contain_text('UI 批量联合实验')
        expect(page.locator('#studyDetail')).to_contain_text('draft')
        expect(page.locator('#studyDetail')).to_contain_text('模型调用为零')
        assert page.locator('#generateAdvice').count()==0
        page.screenshot(path=str(out/'03-batch-matrix.png'),full_page=True)
        page.locator('[data-tab="bi"]').click()
        page.locator('#openDemo').click()
        expect(page.locator('#dataPreview')).to_contain_text('合成演示数据')
        page.locator('#dataBaseline').select_option(label='基准')
        page.locator('#compareData').click()
        expect(page.locator('#analysisResult')).to_contain_text('汇总 3 个实验组')
        expect(page.locator('#analysisResult img').first).to_have_js_property('complete',True)
        assert page.locator('#analysisResult img').first.evaluate('(e)=>e.naturalWidth>0')
        with page.expect_download() as download:
            page.get_by_role('link',name='导出对照表 CSV').click()
        download.value.save_as(str(out/'demo-comparison.csv'))
        page.locator('#dataAuditStatement').fill('电网能耗降低5%。')
        page.locator('.operand-group').nth(0).select_option(label='基准')
        page.locator('.operand-group').nth(1).select_option(label='奖励调整')
        page.locator('#auditData').click()
        expect(page.locator('#dataAuditResult')).to_contain_text('数值一致')
        page.screenshot(path=str(out/'04-demo-analysis.png'),full_page=True)
        # Real CSV UI import, not a substituted response.
        page.get_by_text('导入已有实验 CSV',exact=True).click()
        page.locator('#csvName').fill('UI CSV 导入验证')
        page.locator('#csvFile').set_input_files(str(ROOT/'web/research-data-template.csv'))
        page.locator('#csvSource').fill('合成 CSV 模板，界面回归专用')
        page.locator('#importCSV').click()
        expect(page.locator('#dataPreview')).to_contain_text('UI CSV 导入验证')
        page.locator('#compareData').click()
        expect(page.locator('#analysisResult')).to_contain_text('汇总 2 个实验组')
        page.set_viewport_size({'width':390,'height':844})
        for tab in ('lab','bi'):
            page.locator(f'[data-tab="{tab}"]').click()
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),tab
            page.screenshot(path=str(out/f'{tab}-mobile.png'),full_page=True)
        browser.close()
    (out/'result.json').write_text(json.dumps({'errors':errors,'checks':['matrix preview 4 configurations / 12 executions','confirmation invalidation','frozen draft only','no advice controls','independent demo','CSV import','multi metric table','SVG','CSV export','draft number audit','desktop and mobile'],'url':args.url},ensure_ascii=False,indent=2),encoding='utf-8')
    assert not errors,errors
    print('Research workspace UI checks passed')


if __name__=='__main__': main()
