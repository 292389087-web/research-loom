'use strict';
const $ = id => document.getElementById(id);
const state = {papers: [], selected: new Set(), mode: 'understand', run: null, source: null, result: null, tab: 'all', viewer: null, reviewItem: null, eventIds: new Set()};
const categories = {problem:'研究问题',method:'方法与架构',contribution:'核心贡献',metric:'评价指标',dataset:'数据与协议',baseline:'对比基线',ablation:'消融实验',limitation:'局限与边界',figure:'图表理解'};
const modes = {understand:'论文理解',plan:'研究方向与实验方案',question:'带证据追问',figure:'图与架构理解'};
const statuses = {queued:'排队中',running:'进行中',completed:'已完成',failed:'失败',cancelled:'已取消',ready:'可研究',parsing:'解析中'};
const escapeHtml = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const e = escapeHtml;
async function api(url, options={}) {
  const response = await fetch(url, options);
  if (!response.ok) { let message; try { const data=await response.json(); message=typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail); } catch { message=response.statusText; } throw new Error(message); }
  return response.json();
}
const post = (url, body) => api(url, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
let toastTimer;
function toast(message) { $('toast').textContent=message; $('toast').classList.remove('hidden'); clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').classList.add('hidden'),6000); }
async function refresh() {
  try {
    const [papers,runs,status]=await Promise.all([api('/api/papers'),api('/api/runs'),api('/api/status')]);
    state.papers=papers; renderPapers(); renderHistory(runs);
    $('modelStatus').textContent=status.model_available ? `${status.worker_embedded===false?'独立执行进程':status.model_loaded?'模型已加载':'模型就绪，按需加载'} · ${status.queue_size} 个任务` : '未找到权重，请检查配置';
  } catch(error) { toast(error.message); }
}
function renderPapers() {
  $('paperCount').textContent=state.papers.length;
  $('paperList').innerHTML=state.papers.map(p=>`<div class="paper-card ${state.selected.has(p.id)?'selected':''}"><input type="checkbox" aria-label="选择 ${e(p.title)}" data-paper="${e(p.id)}" ${state.selected.has(p.id)?'checked':''} ${p.status!=='ready'?'disabled':''}><div class="paper-info"><strong title="${e(p.title)}">${e(p.title)}</strong><div class="paper-meta"><span>${p.pages} 页 · ${statuses[p.status]||e(p.status)}</span><button data-preview="${e(p.id)}">原文 ↗</button>${p.status!=='ready'?`<button data-resume="${e(p.id)}">继续解析</button>`:''}</div></div></div>`).join('');
  $('selectionCount').textContent=state.selected.size ? `已选择 ${state.selected.size} 篇论文` : '尚未选择论文';
}
function renderHistory(runs) {
  $('history').innerHTML=runs.filter(r=>r.kind==='research').map(r=>`<button class="history-item" data-run="${r.id}" title="${e(r.request.question||modes[r.request.mode])}">${e(r.request.question||modes[r.request.mode])}<span>${statuses[r.status]} · ${new Date(r.created).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'})}</span></button>`).join('') || '<p class="empty-note">研究记录会自动保存在这里</p>';
}
function setMode(mode) {
  state.mode=mode;
  document.querySelectorAll('.mode').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));
  $('figureControls').classList.toggle('hidden',mode!=='figure');
  $('followupHint').textContent=mode==='question'&&state.result ? '继续讨论当前研究记录，保留上一轮上下文' : '所选论文是本次研究的证据范围';
}
async function uploadFiles(files) {
  for (const file of [...files]) {
    try {
      $('uploadStatus').textContent=`正在上传 ${file.name}…`;
      const data=new FormData(); data.append('file',file);
      const response=await api('/api/papers',{method:'POST',body:data});
      $('uploadStatus').textContent=response.duplicate?'相同论文已存在，沿用解析结果':'文件已保存，正在解析页面…';
      state.selected.add(response.paper.id);
      await refresh();
      if(response.run_id) watch(response.run_id);
      else $('uploadStatus').textContent='论文已就绪';
    } catch(error) { $('uploadStatus').textContent=error.message; toast(error.message); }
  }
}
function getProfile() {
  return {interests:$('interests').value,experience:$('experience').value,compute:$('compute').value,timeline:$('timeline').value||'尚未确定',goal:$('question').value||'理解方法，寻找可验证的研究切入点'};
}
async function startRun() {
  const papers=[...state.selected].filter(id=>state.papers.some(p=>p.id===id&&p.status==='ready'));
  if(!papers.length) return toast('请先上传并选择已解析的论文');
  if(papers.length!==state.selected.size) return toast('部分所选论文尚未解析完成，请等待或取消选择');
  const body={paper_ids:papers,mode:state.mode,question:$('question').value,profile:getProfile(),model_policy:$('modelPolicy').value,allow_remote:$('allowRemote').checked,allow_remote_images:$('allowRemoteImages').checked};
  if(state.mode==='figure') {
    body.page=Number($('figurePage').value);
    body.bbox={full:[0,0,1,1],top:[0,0,1,.58],bottom:[0,.42,1,1]}[$('figureRegion').value];
  }
  if(state.mode==='question'&&state.result&&state.run?.status==='completed'&&state.run?.kind==='research') body.parent_run_id=state.run.id;
  $('startRun').disabled=true;
  try {
    localStorage.setItem('loom-profile',JSON.stringify(body.profile));
    const serialized=JSON.stringify(body);
    let pending; try { pending=JSON.parse(sessionStorage.getItem('loom-pending-submit')); } catch {}
    if(!pending||pending.body!==serialized) pending={body:serialized,key:crypto.randomUUID()};
    sessionStorage.setItem('loom-pending-submit',JSON.stringify(pending));
    const response=await api('/api/runs',{method:'POST',headers:{'Content-Type':'application/json','Idempotency-Key':pending.key},body:serialized});
    sessionStorage.removeItem('loom-pending-submit');
    state.result=null; $('results').classList.add('hidden'); $('welcome').classList.add('hidden'); watch(response.run_id); await refresh();
  }
  catch(error){toast(error.message);}
  finally{$('startRun').disabled=false;}
}
function watch(runId) {
  if(state.source) state.source.close();
  state.eventIds.clear(); $('eventList').innerHTML='';
  state.run={id:runId}; $('progressPanel').classList.remove('hidden','failed'); $('progressTitle').textContent='研究进行中';
  $('traceLink').href=`/api/runs/${runId}/trace`;
  $('cancelRun').classList.remove('hidden'); $('retryRun').classList.add('hidden');
  const source=new EventSource(`/api/runs/${runId}/events`); state.source=source;
  source.addEventListener('progress',message=>{
    if(state.source!==source)return;
    const event=JSON.parse(message.data); if(state.eventIds.has(event.seq))return; state.eventIds.add(event.seq);
    const text=event.payload.message||event.type; $('latestEvent').textContent=text;
    const item=document.createElement('li'); item.textContent=`${new Date(event.time).toLocaleTimeString('zh-CN')} · ${text}`; $('eventList').appendChild(item);
    if(event.type==='page_parsed'){$('uploadStatus').textContent=text;}
  });
  source.addEventListener('done',async()=>{source.close(); if(state.source!==source)return; await loadRun(runId); await refresh();});
  source.onerror=()=>{if(state.source===source)$('latestEvent').textContent='连接暂时断开，正在恢复进度；任务仍保存在本地';};
}
async function loadRun(runId) {
  try {
    const run=await api(`/api/runs/${runId}`); state.run=run;
    $('traceLink').href=`/api/runs/${runId}/trace`;
    if(run.status==='queued'||run.status==='running'){watch(runId);return;}
    if(state.source)state.source.close();
    $('progressTitle').textContent=run.kind==='ingest'?'论文解析'+(statuses[run.status]||''):(statuses[run.status]||'');
    $('progressPanel').classList.remove('hidden'); $('progressPanel').classList.toggle('failed',run.status==='failed');
    $('cancelRun').classList.add('hidden'); $('retryRun').classList.toggle('hidden',!['failed','cancelled'].includes(run.status));
    $('latestEvent').textContent=run.error || (run.status==='completed'?'结果已保存，可随时返回或继续研究':'已停止，可从已完成阶段重试');
    if(run.kind==='ingest') { $('uploadStatus').textContent=run.status==='completed'?'论文解析完成，可以开始研究':run.error||'解析已停止'; return; }
    if(run.result) {
      state.result=run.result; state.selected=new Set(run.request.paper_ids); state.tab=run.result.directions.length?'directions':'all'; renderPapers(); renderResults();
      $('question').value=run.request.question||''; setMode(run.request.mode);
      for(const key of ['interests','experience','compute','timeline']) $(key).value=run.request.profile[key]||'';
    }
  }catch(error){toast(error.message);}
}
function renderResults() {
  const r=state.result; if(!r)return;
  $('welcome').classList.add('hidden'); $('results').classList.remove('hidden');
  $('resultTitle').textContent=modes[r.mode]||'研究笔记';
  $('resultScope').textContent=`基于 ${r.scope.paper_ids.length} 篇本地论文。${r.scope.novelty}`;
  const seconds=r.usage.reduce((sum,u)=>sum+u.seconds,0);
  $('resultStats').textContent=`${r.findings.length} 条研究发现 · ${r.quality.quotes_matched} 条原文摘录匹配 · ${r.quality.semantic_review_flags??r.quality.needs_review} 条结论需注意 · 模型用时 ${seconds.toFixed(1)} 秒 · ${r.retrieval.mode}`;
  $('exportMd').href=`/api/runs/${state.run.id}/export`; $('exportJson').href=`/api/runs/${state.run.id}/export?format=json`;
  const tabs=[['all','全部发现'],['method','问题与方法'],['experiment','实验与指标'],['limits','局限与图表']]; if(r.directions.length)tabs.unshift(['directions','研究切入点']);
  $('resultTabs').innerHTML=tabs.map(([id,label])=>`<button class="tab ${state.tab===id?'active':''}" data-tab="${id}">${label}</button>`).join('');
  renderContent();
  $('openQuestions').innerHTML=r.open_questions.length?`<div class="open-questions"><h3>进一步规划前，需要明确</h3><ul>${r.open_questions.map(q=>`<li>${e(q)}</li>`).join('')}</ul><button class="secondary" id="continueResearch">补充背景，继续讨论 ↗</button></div>`:'';
}
function citationButton(c) {
  const ev=state.result.evidence.find(v=>v.id===c.evidence_id);
  const paper=ev&&state.papers.find(p=>p.id===ev.paper_id);
  return `<button class="citation ${!c.quote_matches?'invalid':''}" data-cite="${e(c.evidence_id)}" title="${e(c.quote||'')}">${ev?`${e(paper?.title?.slice(0,22)||'论文')} · p.${ev.page}`:'引用待核查'} ↗</button>`;
}
function reviewHTML(item) {
  const reviews=(state.run.reviews||[]).filter(r=>r.item_id===item.id); const last=reviews.at(-1);
  const check=item.model_review?.recommendation ? `<div class="plan-review"><span class="tag amber">方案复核：${item.model_review.recommendation==='revise_first'?'建议先修改':'可在核查后探索'}</span><ul>${item.model_review.concerns.map(c=>`<li>${e(c)}</li>`).join('')}</ul><p><b>先做这项验证：</b>${e(item.model_review.next_check)}</p><small>同一模型的辅助审查，不构成可行性保证。</small></div>` : '';
  return `${check}<div class="finding-bottom"><span class="review-note">${last?`人工复核：${last.verdict==='accepted'?'可采用':'需修改'}${last.note?' · '+e(last.note):''}`:''}</span><button class="review-button" data-review="${item.id}">记录人工复核 ↗</button></div>`;
}
function renderContent() {
  const r=state.result;
  if(state.tab==='directions') {
    const labels={hypothesis:'可检验假设',setup:'实验设置',baselines:'对比基线',metrics:'评价指标',ablations:'消融设计',falsification:'什么结果会否定假设',feasibility:'资源与可行性'};
    $('resultContent').innerHTML=r.directions.map(d=>`<article class="direction"><span class="tag amber">研究假设 · 新颖性待核查</span><h3>${e(d.title)}</h3><p>${e(d.motivation)}</p><h4>具体改变</h4><p>${e(d.change)}</p><div class="citations">${d.evidence_ids.map(id=>citationButton({evidence_id:id,quote_matches:true})).join('')}</div><h4>成立所需的假设</h4><ul>${d.assumptions.map(a=>`<li>${e(a)}</li>`).join('')}</ul><div class="experiment-grid">${Object.entries(d.experiment).filter(([key])=>Object.hasOwn(labels,key)).map(([key,value])=>`<div class="experiment-cell ${['hypothesis','falsification','feasibility'].includes(key)?'wide':''}"><b>${labels[key]}</b><p>${e(Array.isArray(value)?value.map(v=>'• '+v).join('\n'):value)}</p></div>`).join('')}</div><h4>需要进一步检索的相关工作</h4><div class="queries">${d.novelty_queries.map(e).join('\n')}</div><h4>主要风险</h4><ul>${d.risks.map(v=>`<li>${e(v)}</li>`).join('')}</ul>${reviewHTML(d)}</article>`).join(''); return;
  }
  const groups={method:['problem','method','contribution'],experiment:['metric','dataset','baseline','ablation'],limits:['limitation','figure']};
  const items=r.findings.filter(f=>state.tab==='all'||groups[state.tab]?.includes(f.category));
  $('resultContent').innerHTML=items.map(f=>`<article class="finding"><div class="finding-top"><span class="tag">${categories[f.category]}</span><span class="tag ${f.source_status==='quote_matched'?'gray':'amber'}">${{quote_matched:'摘录匹配 · 语义待核实',needs_review:'引用待复核',visual_interpretation:'视觉解读 · 待复核'}[f.source_status]}</span>${f.basis==='model_inference'?'<span class="tag amber">模型推断</span>':''}<span class="finding-id">${f.id.toUpperCase()}</span></div><p>${e(f.statement)}</p><div class="citations">${f.citations.map(citationButton).join('')}</div>${f.research_implication?`<div class="implication"><b>对研究设计的启发 · 模型建议</b><p>${e(f.research_implication)}</p></div>`:''}${f.model_review?`<div class="model-review"><span>模型复核：${{supported:'引文支持',partial:'仅部分支持',unsupported:'未获支持',uncertain:'尚不确定'}[f.model_review.status]}</span> ${e(f.model_review.reason)}<small>同一模型复核，仍需人工判断</small></div>`:''}${(f.review_reasons||[]).map(reason=>`<p class="review-warning">${e(reason)}</p>`).join('')}${reviewHTML(f)}</article>`).join('')||'<p class="empty-note">当前结果中没有这一类发现。可针对该问题继续追问。</p>';
}
async function showPage(paperId,page=1,ev=null) {
  try {
    const paper=state.papers.find(p=>p.id===paperId)||await api(`/api/papers/${paperId}`);
    state.viewer={paperId,page,pages:paper.pages}; $('viewer').classList.remove('hidden');
    $('viewerTitle').textContent=paper.title; $('viewerPage').textContent=`${page} / ${paper.pages}`;
    $('pageImage').src=`/api/papers/${paperId}/pages/${page}/image`;
    $('openPdf').href=`/api/papers/${paperId}/pdf#page=${page}`;
    $('prevPage').disabled=page<=1; $('nextPage').disabled=page>=paper.pages;
    $('quoteText').textContent=ev?.content||'查看原文，或让模型解释当前页面的框架与实验图。';
    $('evidenceType').textContent=ev?.kind==='visual'?'模型视觉解读（请对照原图核实）':'原文片段';
    $('evidenceHighlight').classList.toggle('hidden',!ev);
    if(ev){const [x0,y0,x1,y1]=ev.bbox; Object.assign($('evidenceHighlight').style,{left:x0*100+'%',top:y0*100+'%',width:(x1-x0)*100+'%',height:(y1-y0)*100+'%'});}
  } catch(error){toast(error.message);}
}
async function showEvidence(id){try{const ev=state.result?.evidence.find(e=>e.id===id)||await api(`/api/evidence/${id}`);await showPage(ev.paper_id,ev.page,ev);}catch(error){toast(error.message);}}
document.addEventListener('click',async event=>{
  const button=event.target.closest('button'); if(!button)return;
  if(button.dataset.mode)setMode(button.dataset.mode);
  if(button.dataset.prompt)$('question').value=button.dataset.prompt;
  if(button.dataset.preview)showPage(button.dataset.preview);
  if(button.dataset.resume){try{const response=await post(`/api/papers/${button.dataset.resume}/resume`,{});watch(response.run_id);}catch(error){toast(error.message);}}
  if(button.dataset.run)loadRun(button.dataset.run);
  if(button.dataset.tab){state.tab=button.dataset.tab;renderResults();}
  if(button.dataset.cite)showEvidence(button.dataset.cite);
  if(button.dataset.review){state.reviewItem=button.dataset.review;$('reviewNote').value='';$('reviewDialog').showModal();}
  if(button.id==='continueResearch'){setMode('question');$('question').value='';$('question').focus();$('question').scrollIntoView({behavior:'smooth',block:'center'});}
});
$('paperList').addEventListener('change',event=>{const id=event.target.dataset.paper;if(id){event.target.checked?state.selected.add(id):state.selected.delete(id);renderPapers();}});
$('fileInput').onchange=event=>{uploadFiles(event.target.files);event.target.value='';};
for(const kind of ['dragenter','dragover'])$('dropZone').addEventListener(kind,event=>{event.preventDefault();$('dropZone').classList.add('dragging');});
for(const kind of ['dragleave','drop'])$('dropZone').addEventListener(kind,event=>{event.preventDefault();$('dropZone').classList.remove('dragging');if(kind==='drop')uploadFiles(event.dataTransfer.files);});
$('startRun').onclick=startRun;
$('refreshHistory').onclick=refresh;
$('cancelRun').onclick=async()=>{try{if(state.run)await post(`/api/runs/${state.run.id}/cancel`,{});}catch(error){toast(error.message);}};
$('retryRun').onclick=async()=>{try{const response=await post(`/api/runs/${state.run.id}/retry`,{});watch(response.run_id);}catch(error){toast(error.message);}};
$('closeViewer').onclick=()=>$('viewer').classList.add('hidden');
$('prevPage').onclick=()=>showPage(state.viewer.paperId,state.viewer.page-1);
$('nextPage').onclick=()=>showPage(state.viewer.paperId,state.viewer.page+1);
$('readThisPage').onclick=()=>{state.selected=new Set([state.viewer.paperId]);renderPapers();setMode('figure');$('figurePage').value=state.viewer.page;$('question').value='请解释这一页的框架、模块关系或实验图，结合正文指出它支持什么结论、有哪些不确定信息。';$('viewer').classList.add('hidden');$('question').focus();toast('已选好页面，可调整问题或阅读区域后开始研究');};
$('saveReview').onclick=async()=>{try{await post(`/api/runs/${state.run.id}/reviews`,{item_id:state.reviewItem,verdict:$('reviewVerdict').value,note:$('reviewNote').value});$('reviewDialog').close();state.run=await api(`/api/runs/${state.run.id}`);renderContent();toast('复核记录已保存');}catch(error){toast(error.message);}};
$('newWorkspace').onclick=()=>{state.source?.close();state.result=null;state.run=null;$('results').classList.add('hidden');$('progressPanel').classList.add('hidden');$('welcome').classList.remove('hidden');$('question').value='';setMode('understand');};
try{const profile=JSON.parse(localStorage.getItem('loom-profile')||'{}');for(const key of ['interests','experience','compute','timeline'])if(profile[key])$(key).value=profile[key];}catch{}
refresh();
setInterval(refresh,15000);
