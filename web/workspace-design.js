/* Deterministic batch experiments and independent research data analysis. */
let studyRows=[],dataSet=null,lastReport=null,demoMode=false,batchPreview=null;
const groupNames={reward:'奖励函数子项系数',hyperparameter:'模型与训练超参数',protocol:'实验协议参数',unclassified:'其他登记参数'};
const adapterNow=()=>adapters.find(a=>a.id===$('trainer').value);
const metricLabel=(a,k)=>a?.metric_metadata?.[k]?.label||k;
function selectedMetrics(id){return [...$(id).querySelectorAll('input:checked')].map(e=>e.value);}
function metricChoices(id,metrics,selected,adapter){
  $(id).innerHTML=metrics.map(k=>`<label class="check"><input type="checkbox" value="${esc(k)}" ${selected.includes(k)?'checked':''}>${esc(metricLabel(adapter,k))}<small>${esc(adapter?.metric_metadata?.[k]?.unit||'')}</small></label>`).join('');
}
function ruleRow(rule={}){
  const a=adapterNow(), div=document.createElement('div');div.className='row metric-rule';
  div.innerHTML=`<label>指标<select class="rule-metric">${(a?.metrics||[]).map(k=>`<option value="${esc(k)}" ${rule.metric===k?'selected':''}>${esc(metricLabel(a,k))}</option>`).join('')}</select></label><label>条件<select class="rule-op"><option value="le">不高于</option><option value="ge" ${rule.operator==='ge'?'selected':''}>不低于</option></select></label><label>阈值<input class="rule-value" type="number" step="any" value="${esc(rule.value??0)}"></label><button type="button">移除</button>`;
  div.querySelector('button').onclick=()=>div.remove();$('metricRules').append(div);
}
const getRules=()=>[...$('metricRules').children].map(e=>({metric:e.querySelector('.rule-metric').value,operator:e.querySelector('.rule-op').value,value:Number(e.querySelector('.rule-value').value)}));

function invalidateBatch(){batchPreview=null;$('confirmBatch').checked=false;$('createStudy').disabled=true;$('batchPreview').textContent='配置已更改，请重新预览完整矩阵。';}
function readCandidates(){
  const params={};
  for(const el of $('parameterGroups').querySelectorAll('[data-param]')){
    const parts=el.value.trim().split(/[,，\s]+/);
    if(!el.value.trim()||parts.some(x=>!x||!Number.isFinite(Number(x))))throw Error(el.dataset.param+' 需要有限数值');
    params[el.dataset.param]=parts.map(Number);
  }
  $('parameters').value=JSON.stringify(params);return params;
}
function readExplicitRows(){
  let rows;
  try{rows=JSON.parse($('explicitRows').value);}catch(e){throw Error('JSON 格式错误：请使用英文双引号、逗号和方括号，不要写注释或末尾多余逗号。可重新生成模板后修改数值。');}
  const candidates=readCandidates(),keys=Object.keys(candidates),seen=new Set();
  if(!Array.isArray(rows)||!rows.length||rows.length>100)throw Error('JSON 顶层必须是含 1 至 100 组配置的数组。');
  rows.forEach((row,i)=>{
    if(!row||Array.isArray(row)||typeof row!=='object')throw Error(`第 ${i+1} 组必须是一个 JSON 对象。`);
    const missing=keys.filter(k=>!Object.hasOwn(row,k)),extra=Object.keys(row).filter(k=>!keys.includes(k));
    if(missing.length||extra.length)throw Error(`第 ${i+1} 组：缺少字段 ${missing.join(', ')||'无'}；未登记字段 ${extra.join(', ')||'无'}。请保留模板中的完整键名。`);
    for(const k of keys)if(typeof row[k]!=='number'||!Number.isFinite(row[k])||!candidates[k].includes(row[k]))throw Error(`第 ${i+1} 组 ${k} 必须是上方候选集中的数值（不加引号）。`);
    const signature=JSON.stringify(keys.map(k=>row[k]));if(seen.has(signature))throw Error(`第 ${i+1} 组配置重复，请修改或删除。`);seen.add(signature);
  });return rows;
}
function generateExplicitTemplate(){
  const candidates=readCandidates();if(!Object.keys(candidates).length)throw Error('请先选择训练适配器。');
  $('explicitRows').value=JSON.stringify([Object.fromEntries(Object.entries(candidates).map(([k,v])=>[k,v[0]]))],null,2);
  $('explicitJSONStatus').textContent='已生成一组完整配置。可修改数值或添加不同的配置对象；重新生成会替换本框内容。';invalidateBatch();
}
function readSeeds(){
  const text=$('seeds').value.trim();if(!text)throw Error('随机种子至少填写一个整数，例如 11；可保留默认的 11,23,37。');
  const parts=text.split(/[,，\s]+/);if(parts.some(x=>!x||!Number.isSafeInteger(Number(x))))throw Error('随机种子必须是整数，使用逗号分隔。');
  const values=parts.map(Number);if(values.length>10||new Set(values).size!==values.length)throw Error('请填写 1 至 10 个不重复的随机种子。');return values;
}
function batchRequest(){return {batch_mode:$('batchMode').value,configurations:$('batchMode').value==='explicit'?readExplicitRows():[],
  project_name:$('projectName').value,name:$('studyName').value,trainer_id:$('trainer').value,parameters:readCandidates(),
  protocol:$('protocol').value,seeds:readSeeds(),evaluation_metrics:selectedMetrics('evaluationMetrics'),metric_rules:getRules(),
  objective:$('objective').value||null,direction:$('direction').value,budget_seconds:Number($('budget').value),trial_timeout:Number($('timeout').value),
  failure_policy:$('failurePolicy').value,source_run_id:$('sourceRun').value||null,allow_remote:$('studyRemote').checked};}
trainerChanged=function(){
  const a=adapterNow();
  $('trainerHint').textContent=a?(a.fixture?'工程测试适配器，不能作为真实实验成绩。':'每个配置/seed 在独立进程和目录中执行已登记代码；不调用模型。'):'暂无训练适配器，可先在 04 使用 CSV 或合成演示。';
  $('parameterGroups').innerHTML=Object.entries(groupNames).map(([g,label])=>{
    const keys=Object.keys(a?.parameters||{}).filter(k=>(a.parameter_groups?.[k]||'unclassified')===g);
    return keys.length?`<fieldset><legend>${label}</legend>${keys.map(k=>`<label>${esc(a.parameter_labels?.[k]||k)}<small>${esc(k)} · 已登记：${esc(a.parameters[k].join(', '))}</small><input data-param="${esc(k)}" value="${a.parameters[k][0]}" placeholder="单值固定，多个值用逗号分隔"></label>`).join('')}</fieldset>`:'';
  }).join('');
  $('rangeParameter').innerHTML=Object.keys(a?.parameters||{}).map(k=>`<option value="${esc(k)}">${esc(a.parameter_labels?.[k]||k)}</option>`).join('');
  $('objective').innerHTML='<option value="">不设置（多指标观察）</option>'+(a?.metrics||[]).map(k=>`<option value="${esc(k)}">${esc(metricLabel(a,k))}</option>`).join('');
  metricChoices('evaluationMetrics',a?.metrics||[],(a?.metrics||[]).filter(k=>a.metric_metadata?.[k]?.role==='outcome').slice(0,4),a);
  $('metricRules').innerHTML='';$('explicitRows').value='';$('explicitJSONStatus').textContent='';if($('batchMode').value==='explicit'&&a)generateExplicitTemplate();showMetricHint();invalidateBatch();
};
$('trainer').onchange=trainerChanged;
function showMetricHint(){$('metricHint').textContent=adapterNow()?.metric_metadata?.[$('objective').value]?.description||'排序仅用于查看结果。不同奖励系数的总 reward 不可直接判断优劣。';}
$('objective').onchange=showMetricHint;
$('addMetricRule').onclick=()=>{ruleRow();invalidateBatch();};
$('batchMode').onchange=()=>{try{$('explicitPanel').hidden=$('batchMode').value!=='explicit';if($('batchMode').value==='explicit'&&!$('explicitRows').value.trim())generateExplicitTemplate();invalidateBatch();}catch(e){$('message').textContent=e.message;}};
action('generateJSONTemplate',async()=>generateExplicitTemplate());
action('formatExplicitJSON',async()=>{const rows=readExplicitRows();$('explicitRows').value=JSON.stringify(rows,null,2);$('explicitJSONStatus').textContent=`检查通过：${rows.length} 组完整配置；启动前仍需预览矩阵并确认。`;invalidateBatch();});
$('explicitRows').addEventListener('input',()=>{$('explicitJSONStatus').textContent='内容已修改，请检查格式并重新预览。';});
$('lab').addEventListener('input',e=>{if(e.target.closest('#lab .columns article')?.contains($('previewBatch'))&&e.target.id!=='confirmBatch')invalidateBatch();});
$('metricRules').addEventListener('click',invalidateBatch);
$('confirmBatch').onchange=()=>{$('createStudy').disabled=!batchPreview||!$('confirmBatch').checked;};
action('fillRange',async()=>{
  const start=Number($('rangeStart').value),end=Number($('rangeEnd').value),step=Number($('rangeStep').value);
  if(![start,end,step].every(Number.isFinite)||step<=0||end<start||Math.floor((end-start)/step)>99)throw Error('范围需递增、步长大于零，最多 100 个值');
  const values=[];for(let n=0;n<100;n++){const v=Number((start+n*step).toPrecision(12));if(v>end+Math.abs(step)*1e-9)break;values.push(v);}
  const el=[...$('parameterGroups').querySelectorAll('[data-param]')].find(e=>e.dataset.param===$('rangeParameter').value);
  if(!el)throw Error('先选择训练适配器');el.value=values.join(', ');invalidateBatch();
});
action('previewBatch',async()=>{
  const request=batchRequest(), result=await api('/api/batches/preview',request);
  let pending;try{pending=JSON.parse(sessionStorage.getItem('loom-batch-submission')||'null');}catch{}
  const key=pending?.hash===result.confirmation_sha256?pending.key:crypto.randomUUID();
  sessionStorage.setItem('loom-batch-submission',JSON.stringify({hash:result.confirmation_sha256,key}));
  batchPreview={request,signature:JSON.stringify(request),result,key};$('confirmBatch').checked=false;$('createStudy').disabled=true;
  $('batchPreview').innerHTML=`<strong>${result.configurations} 组配置 × ${request.seeds.length} 个 seed = ${result.executions} 次训练</strong><p>变化字段：${esc(result.varying_parameters.join('、')||'无（固定配置）')}</p><p>${esc(result.notice)}</p><p>按每次超时上限累计：${(result.timeout_envelope_seconds/3600).toFixed(2)} 小时；不是预计耗时。${result.timeout_envelope_seconds>request.budget_seconds?'总预算小于此上界，可能在完成全部组合前到期。':''}</p>`+table(result.matrix.map((r,i)=>({顺序:i+1,...r})))+`<small>矩阵 SHA-256：${esc(result.matrix_sha256)}</small>`;
});
action('createStudy',async()=>{
  if(!batchPreview||!$('confirmBatch').checked||batchPreview.signature!==JSON.stringify(batchRequest()))throw Error('请重新预览并确认当前配置');
  study=await api('/api/batches',{spec:batchPreview.request,confirmation_sha256:batchPreview.result.confirmation_sha256},'POST',{'Idempotency-Key':batchPreview.key});
  sessionStorage.removeItem('loom-batch-submission');
  invalidateBatch();await loadStudies();await loadStudy(study.id);
});
const oldLoadStudies=loadStudies;
loadStudies=async function(){await oldLoadStudies();studyRows=await api('/api/studies');refreshDataStudies();};
const oldLoadStudy=loadStudy;
loadStudy=async function(id){await oldLoadStudy(id);
  $('pauseStudy').disabled=!study?.spec.batch_mode||study.status!=='running';
  $('resumeStudy').disabled=demoMode||!study?.spec.batch_mode||study.status!=='paused';
  if(!study){$('analyzeBatch').disabled=true;$('artifactAttempt').innerHTML='';return;}
  const attempts=study.trials.flatMap(t=>t.repeats);$('artifactAttempt').innerHTML='<option value="">选择验收失败的试验</option>'+attempts.filter(r=>r.status==='failed'&&r.execution_state==='completed').map(r=>`<option value="${esc(r.id)}">${esc(r.id)} · seed ${r.seed}</option>`).join('');
  $('studyDetail').insertAdjacentHTML('beforeend',table(attempts.map(r=>({试验:r.id,seed:r.seed,训练:r.execution_state||'历史记录',产物验收:r.artifact_state||'历史记录',状态:r.status}))));
  $('analyzeBatch').disabled=false;$('stopStudy').disabled=!['draft','running','pausing','paused'].includes(study.status);
  if(demoMode)$('startStudy').disabled=true;
  $('exportBatchJSON').href=`/api/studies/${study.id}/export?format=json`;
  const counts={};for(const t of study.trials)for(const r of t.repeats)counts[r.status]=(counts[r.status]||0)+1;
  $('studyDetail').insertAdjacentHTML('beforeend',`<p>完成 ${counts.completed||0} · 失败 ${counts.failed||0} · 运行 ${counts.running||0} · 已排队 ${counts.queued||0} · 取消 ${counts.cancelled||0}</p><p>${study.spec.batch_mode?'批次模式：'+esc(study.spec.batch_mode)+'；执行阶段模型调用为零。':'历史单轮计划；原始记录保留。'}</p>`);
  if(study.manifest.batch)$('studyDetail').insertAdjacentHTML('beforeend',`<details><summary>查看全部 ${study.manifest.batch.configurations} 组冻结配置（含尚未执行项）</summary>${table(study.manifest.batch.matrix.map((r,i)=>({顺序:i+1,...r})))}</details>`);
};
action('pauseStudy',async()=>{await api(`/api/studies/${requireStudy()}/pause`,{});await loadStudy(study.id);});
action('resumeStudy',async()=>{await api(`/api/studies/${requireStudy()}/resume`,{});await loadStudy(study.id);});
action('refreshStudy',()=>loadStudy(requireStudy()));
action('analyzeBatch',async()=>{renderDataSet(await api('/api/data/from-studies',{name:study.spec.name,study_ids:[requireStudy()]}));await loadDataSets();document.querySelector('[data-tab="bi"]').click();});

function refreshDataStudies(){
  const selected=$('dataProject').value;const names=[...new Set(studyRows.map(s=>s.spec.project_name||'我的研究'))];
  $('dataProject').innerHTML='<option value="">全部项目（仍检查可比性）</option>'+names.map(n=>`<option>${esc(n)}</option>`).join('');$('dataProject').value=selected;
  $('dataStudyChoices').innerHTML=studyRows.filter(s=>!selected||(s.spec.project_name||'我的研究')===selected).map(s=>`<label class="check"><input type="checkbox" value="${esc(s.id)}">${esc(s.spec.name)} · ${esc(s.status)}</label>`).join('')||'<p>暂无平台实验，可导入 CSV 或打开演示数据。</p>';
}
$('dataProject').onchange=refreshDataStudies;
async function loadDataSets(){const rows=await api('/api/data/datasets');$('dataSetList').innerHTML='<option value="">选择已有数据集</option>'+rows.map(r=>`<option value="${esc(r.id)}">${esc(r.name)}</option>`).join('');if(dataSet)$('dataSetList').value=dataSet.id;}
function renderDataSet(ds){
  dataSet=ds;lastReport=null;$('analysisResult').textContent='数据已选择。生成对照表，或输入分析问题。';$('dataAuditResult').textContent='';
  const gs=[...new Map(ds.rows.map(r=>[r.group,r.label||r.group])).entries()];
  $('dataPreview').innerHTML=`<strong>${esc(ds.name)}</strong><p>${esc(ds.notice)}</p><p>${gs.length} 个实验组 · ${Object.keys(ds.metrics).length} 个指标 · ${ds.rows.length} 条记录 · ${ds.excluded.length} 条未完成记录 · ${(ds.series||[]).length} 条逐日记录</p><p>这是不可变快照，后续完成的实验需点击“创建最新快照”后才会进入分析。</p>`+table(ds.rows.slice(0,6).map(r=>({实验组:r.label||r.group,seed:r.seed,指标:r.metric,数值:r.value,单位:r.unit,阶段:r.phase})))+`<details><summary>来源与快照</summary><p>${esc(ds.source_note)}</p><code>${esc(ds.snapshot)}</code></details>`;
  $('parameterFilters').innerHTML='';$('dataMode').value='summary';$('dayStart').value='';$('dayEnd').value='';refreshDataMode();
  $('dataBaseline').innerHTML='<option value="">不设基准（只汇总）</option>'+gs.map(([k,v])=>`<option value="${esc(k)}">${esc(v)}</option>`).join('');
  $('dataSort').innerHTML='<option value="">请选择排序指标</option>'+Object.keys(ds.metrics).map(k=>`<option>${esc(k)}</option>`).join('');
  $('auditOperands').innerHTML='';addOperand();addOperand();
}
const requireData=()=>{if(!dataSet)throw Error('请选择数据集、导入 CSV 或打开演示数据');return dataSet.id;};
$('dataSetList').onchange=async()=>{try{if($('dataSetList').value)renderDataSet(await api('/api/data/datasets/'+$('dataSetList').value));else{dataSet=null;$('dataPreview').textContent='请选择数据集';}}catch(e){$('message').textContent=e.message;}};
action('openDemo',async()=>{renderDataSet(await api('/api/data/demo',{}));await loadDataSets();});
action('snapshotStudies',async()=>{const ids=[...$('dataStudyChoices').querySelectorAll('input:checked')].map(e=>e.value);if(!ids.length)throw Error('请勾选实验轮次');renderDataSet(await api('/api/data/from-studies',{name:($('dataProject').value||'多轮实验')+' · '+new Date().toLocaleDateString(),study_ids:ids}));await loadDataSets();});
$('csvFile').onchange=async()=>{const f=$('csvFile').files[0];if(!f)return;if(f.size>60000){$('message').textContent='CSV 太大，请缩小至 60 KB 内';return;}$('csvText').value=await f.text();};
action('importCSV',async()=>{renderDataSet(await api('/api/data/import',{name:$('csvName').value,csv_text:$('csvText').value,source_note:$('csvSource').value,allow_remote:$('csvRemote').checked}));await loadDataSets();});
const planFields=()=>({metrics:selectedMetrics('dataMetrics'),baseline:$('dataBaseline').value||null,sort_metric:$('dataSort').value||null,direction:$('dataDirection').value||null,filters:readParameterFilters(),day_start:$('dataMode').value==='trajectory'&&$('dayStart').value?Number($('dayStart').value):null,day_end:$('dataMode').value==='trajectory'&&$('dayEnd').value?Number($('dayEnd').value):null});
async function analyzeData(operation){const fields=planFields();if(operation==='failures'){fields.day_start=null;fields.day_end=null;fields.metrics=[];}showAnalysis(await api(`/api/data/datasets/${requireData()}/analyze`,{operation,...fields}));}
action('compareData',()=>analyzeData($('dataMode').value==='trajectory'?'trajectory':'compare'));action('rankData',()=>analyzeData('rank'));action('failureData',()=>analyzeData('failures'));
document.querySelectorAll('.questionExample').forEach(b=>b.onclick=()=>{$('analysisQuestion').value=b.dataset.question;});
action('askAnalysis',async()=>{const r=await api('/api/data/questions',{dataset_id:requireData(),question:$('analysisQuestion').value,...planFields(),...route()});watch(r.run_id);});
showAnalysis=function(r){
  lastReport=r;
  if(r.mode!=='research_data'){$('analysisResult').innerHTML=table(r.rows||[]);return;}
  const rows=r.rows||[];const metrics=[...new Set(rows.map(x=>x.metric))];
  $('analysisResult').innerHTML=`<strong>${esc(r.conclusion)}</strong><p>${esc(r.notice)}</p><small>数据快照 ${esc(r.dataset_snapshot)}</small>`+
    (r.highlights?.length?'<ul>'+r.highlights.map(h=>`<li>${esc(h)}</li>`).join('')+'</ul>':'')+
    (r.status==='needs_clarification'?'<p>请补充评价标准，再次分析。</p>':`<div class="row"><a href="/api/data/reports/${r.id}/export?format=csv">导出对照表 CSV</a><a href="/api/data/reports/${r.id}/export?format=json">导出报告及来源 JSON</a></div>`)+
    table(rows.map(x=>({实验组:x.label,...(x.day!==undefined?{天:x.day}:{}),参数:x.parameters||{},指标:x.metric,单位:x.unit,均值:x.mean,标准差:x.std??'单 seed，无法估计',n:x.n,约束:x.constraints,相对基准差值:x.delta??'—',可比性:x.comparison})))+
    (r.plan.operation==='failures'?table(r.excluded||[]):'')+
    metrics.map(m=>`<div class="chart"><img alt="${esc(m)} 的实验组均值对比图" src="/api/data/reports/${r.id}/chart?metric=${encodeURIComponent(m)}"><a href="/api/data/reports/${r.id}/chart?metric=${encodeURIComponent(m)}" download="comparison.svg">下载 ${esc(m)} 图表 SVG</a></div>`).join('')+
    `<details><summary>查看分析过程、计算口径与来源</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details>`;
};
function addOperand(){if(!dataSet)return;const gs=[...new Map(dataSet.rows.map(r=>[r.group,r.label||r.group])).entries()];const div=document.createElement('div');div.className='row audit-operand';
  div.innerHTML=`<label>实验组<select class="operand-group">${gs.map(([k,v])=>`<option value="${esc(k)}">${esc(v)}</option>`).join('')}</select></label><label>指标<select class="operand-metric">${Object.keys(dataSet.metrics).map(k=>`<option>${esc(k)}</option>`).join('')}</select></label><button>移除</button>`;
  div.querySelector('button').onclick=()=>div.remove();$('auditOperands').append(div);
}
$('addAuditOperand').onclick=addOperand;
$('dataAuditOperation').onchange=()=>{$('dataAuditStatement').placeholder=$('dataAuditOperation').value==='sum'?'各子项合计为……':'第一项为基准，第二项为对照；例如电网能耗降低 5%。';};
action('auditData',async()=>{const operands=[...$('auditOperands').children].map(e=>({group:e.querySelector('.operand-group').value,metric:e.querySelector('.operand-metric').value}));const r=await api('/api/data/audits',{dataset_id:requireData(),operation:$('dataAuditOperation').value,operands,reported:$('dataAuditReported').value,tolerance:$('dataAuditTolerance').value,statement:$('dataAuditStatement').value,draft_id:$('bindDraft').checked?requireDraft():null});
  $('dataAuditResult').innerHTML=`<h3>${r.passed?'数值一致':'发现数值差异'}</h3><p>计算值 ${esc(r.expected)} ${esc(r.unit)} · 初稿值 ${esc(r.reported)} · 差值 ${esc(r.difference)}</p><p>${esc(r.notice_scope)} ${esc(r.notice)}</p><a href="/api/data/reports/${r.id}/export?format=json">导出核查记录</a><details><summary>公式与逐项来源</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details>`;
});
loadDataSets().catch(e=>$('message').textContent=e.message);
api('/api/models').then(m=>{demoMode=!!m.demo_mode;if(demoMode){for(const id of ['askAnalysis','recommend','startStudy','resumeStudy','preflightTrainer'])$(id).disabled=true;$('askAnalysis').textContent='自然语言模型在演示模式下关闭';$('routePolicy').disabled=true;$('routeRemote').disabled=true;$('modelState').textContent='CPU 演示 · 合成数据可直接分析 · 不执行模型或训练';}}).catch(e=>$('message').textContent=e.message);

action('preflightTrainer',async()=>{const r=await api(`/api/trainers/${encodeURIComponent($('trainer').value)}/preflight`);$('preflightResult').innerHTML=`<strong>${r.passed?'依赖预检通过':'依赖预检未通过'}</strong><p>${esc(r.errors.join('；'))}</p><p>${esc(r.scope)}</p>`;});
action('revalidateAttempt',async()=>{const aid=$('artifactAttempt').value;if(!aid)throw Error('请选择有训练完成凭据的验收失败项');const limit=$('artifactLimit').value;const r=await api(`/api/attempts/${aid}/revalidate`,{note:$('artifactNote').value,artifact_max_bytes:limit?Number(limit)*1024*1024:null});$('artifactReviewResult').textContent='重新验收通过，原始文件未改动，未重新训练。';await loadStudy(requireStudy());});
action('refreshDataSnapshot',async()=>{requireData();if(!dataSet.study_ids?.length)throw Error('只有平台实验快照可从原批次更新');renderDataSet(await api('/api/data/from-studies',{name:dataSet.name+' · 更新',study_ids:dataSet.study_ids}));await loadDataSets();});
function refreshDataMode(){if(!dataSet)return;const daily=$('dataMode').value==='trajectory';$('dayRange').hidden=!daily;$('rankData').disabled=daily;$('compareData').textContent=daily?'生成逐日曲线与对照表':'生成多指标对照表';const metrics=Object.keys(daily?(dataSet.series_metrics||{}):dataSet.metrics);metricChoices('dataMetrics',metrics,metrics.slice(0,3),null);if(daily&&!metrics.length)$('message').textContent='此快照未包含已声明的逐日产物；请用新版适配器完成实验并创建快照。';}
$('dataMode').onchange=refreshDataMode;
$('addParameterFilter').onclick=()=>{if(!dataSet)return;const keys=[...new Set([...dataSet.rows,...(dataSet.excluded||[])].flatMap(r=>Object.keys(r.parameters||{})))];if(!keys.length){$('message').textContent='此数据没有参数字段，无法按参数筛选。';return;}const div=document.createElement('div');div.className='row parameter-filter';div.innerHTML=`<label>参数<select class="filter-key">${keys.map(k=>`<option>${esc(k)}</option>`).join('')}</select></label><label>允许值（逗号分隔）<input class="filter-values"></label><button>移除筛选</button>`;const update=()=>{const values=[...new Set([...dataSet.rows,...(dataSet.excluded||[])].map(r=>r.parameters?.[div.querySelector('select').value]).filter(v=>v!==undefined))];div.querySelector('input').value=values.join(', ');};div.querySelector('select').onchange=update;div.querySelector('button').onclick=()=>div.remove();$('parameterFilters').append(div);update();};
function readParameterFilters(){return [...$('parameterFilters').children].map(e=>{const text=e.querySelector('input').value.trim();if(!text)throw Error('筛选值不能为空');const values=text.split(/[,，\s]+/).map(Number);if(values.some(v=>!Number.isFinite(v)))throw Error('筛选值必须是数字');return {parameter:e.querySelector('select').value,values};});}
