/* GNNBGP Industrial Workbench: binds ONLY to the real NEW bgp CLI contract.
   No fabricated bandgap/DFT values. Direct/indirect type only if engine returns it. */
"use strict";
const $=id=>document.getElementById(id);
const esc=t=>String(t??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=(v,d=4)=>typeof v==="number"&&Number.isFinite(v)?v.toFixed(d):"—";
const dt=t=>t?String(t).replace("T"," ").slice(0,19):"—";
// Icons are an offline, local SVG symbol sprite included in index.html.
// iconHTML receives only constant identifiers authored in this application.
const iconHTML=(name,cls="ui-icon")=>`<svg class="${cls}" aria-hidden="true" focusable="false"><use href="#i-${name}"></use></svg>`;
// Windows Explorer-style document icons; only the icon changes, not file parsing.
function fileIconFor(filename) {
  const base=String(filename||"").replace(/\\/g,"/").split("/").pop().toLowerCase();
  if(base==="poscar"||base==="contcar")return "file-poscar";
  if(base.endsWith(".cif"))return "file-cif";
  if(base.endsWith(".vasp"))return "file-vasp";
  if(base.endsWith(".xyz"))return "file-xyz";
  if(base.endsWith(".json"))return "file-json";
  if(base.endsWith(".csv"))return "file-csv";
  if(base.endsWith(".log")||base.endsWith(".out"))return "file-log";
  if(base.endsWith(".pdf"))return "file-pdf";
  if(base.endsWith(".txt"))return "file-text";
  return "file-text";
}

const names={"gnn-hse06":"GNN / HSE06","abacus-hse06":"ABACUS / HSE06","abacus-pbe0":"ABACUS / PBE0","cider-pbe0":"CIDER / PBE0"};
// Source-aligned computing groups: a business category selects one real CLI backend.
const computeGroups={
 ml:{label:"机器学习带隙误差修正",methods:[["gnn-hse06","GNN-HSE06 · MACE 模型预测"]]},
 gpu:{label:"GPU 带隙计算",methods:[["abacus-hse06","ABACUS-HSE06 · GPU 杂化泛函"],["abacus-pbe0","ABACUS-PBE0 · 25% 精确交换"]]},
 functional:{label:"数据驱动泛函带隙计算",methods:[["cider-pbe0","CIDER-PBE0 · 数据驱动非局域泛函"]]}
};
const categoryOfBackend=id=>Object.keys(computeGroups).find(key=>computeGroups[key].methods.some(pair=>pair[0]===id))||"ml";

const translations={success:"完成",failed:"失败",interrupted:"中断",queued:"排队",running:"运行中",stopping:"停止中",insulator:"绝缘体/半导体",zero_gap:"零带隙",metal:"金属",partial_occupancy:"部分占据",unknown:"未知",direct:"直接带隙",indirect:"间接带隙"};
const S={
 materials:[],samples:[],jobs:[],environment:null,selectedMaterial:null,selectedJob:null,view:"structure",
 dock:"output",detail:"settings",logSize:-1,preview:null,
 angleX:-.47,angleZ:.64,zoom:1,dragging:null,restore:false,
 currentLog:"",clearedAt:0,loopBusy:false,menu:null,
 sourceBusy:false,abacusEdited:false
};
function tr(x){return translations[x]||x||"—"}
function toast(msg,kind="info"){let node=document.createElement("div");node.className="toast "+kind;node.textContent=msg;$("toast-stack").append(node);setTimeout(()=>node.remove(),6000)}
function setStatus(msg){$("status-message").textContent=msg;$("toolbar-text").textContent=msg}
async function api(path, options={}){
  const method=options.method||"GET";
  const r=await fetch(path,{...options,headers:{"X-BGP-UI":"desktop",...(options.headers||{})}});
  const raw=await r.text();
  let j;
  try{j=JSON.parse(raw)}catch{throw new Error("服务返回非 JSON: "+raw.slice(0,180))}
  if(!r.ok)throw new Error(j.error||("HTTP "+r.status));
  return j;
}
const send=(url,obj)=>api(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(obj)});
function selectedMaterial(){return S.materials.find(x=>x.id===S.selectedMaterial)||null}
function selectedJob(){return S.jobs.find(x=>x.id===S.selectedJob)||null}
function jobResult(j){const r=j?.result;return r&&r.status==="success"&&j.state==="success"&&r.converged!==false?r:null}
function gapValue(j){const r=jobResult(j),v=r?.prediction?.gap_ev;return typeof v==="number"&&Number.isFinite(v)?v:null}
function formatSize(n){if(n<1024)return n+" B";if(n<1024*1024)return(n/1024).toFixed(1)+" KB";return(n/1024/1024).toFixed(1)+" MB"}
function exportCSV(filename,rows){const csv="\ufeff"+rows.map(r=>r.map(c=>'"'+String(c??"").replace(/"/g,'""')+'"').join(",")).join("\r\n");const link=document.createElement("a");const url=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));link.href=url;link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(url),1500)}
function activateView(view){
 if(!["structure","results","report","jobs","compare","artifacts"].includes(view))return;
 S.view=view;
 document.querySelectorAll(".editor-tab").forEach(e=>e.classList.toggle("active",e.dataset.view===view));
 document.querySelectorAll(".view").forEach(e=>e.classList.toggle("active",e.id==="view-"+view));
 if(view==="compare")renderCompare();
 if(view==="artifacts")loadArtifacts();
 if(view==="results")renderResult();
 if(view==="report")renderReport();
 if(view==="structure")requestAnimationFrame(drawStructure);
 setStatus({structure:"结构视图",results:"计算结果",report:"计算报告",jobs:"任务管理",compare:"数据比较",artifacts:"输出文件"}[view]);
}
function activateDock(dock){
 S.dock=dock;document.querySelectorAll(".dock-tab").forEach(e=>e.classList.toggle("active",e.dataset.dock===dock));
 document.querySelectorAll(".dock-body").forEach(e=>e.classList.toggle("active",e.id==="dock-"+dock));
}
function renderMaterials(){
 const query=$("material-filter").value.trim().toLowerCase();
 const arr=S.materials.filter(m=>(m.label+" "+m.formula).toLowerCase().includes(query));
 $("material-count").textContent=S.materials.length;
 $("materials-list").innerHTML=arr.length?arr.map(m=>`
   <div class="material-tree-row ${S.selectedMaterial===m.id?"selected":""}">
     <button class="tree-item ${S.selectedMaterial===m.id?"selected":""}" data-material="${esc(m.id)}" title="${esc(m.path)}">
       <span class="materialicon">${iconHTML(fileIconFor(m.filename))}</span><span class="tree-name">${esc(m.label)}</span><span class="tree-count">${m.atom_count}</span>
     </button>
     <button type="button" class="tree-delete" data-remove-material="${esc(m.id)}" aria-label="删除 ${esc(m.label)}" title="删除此结构（保留已有计算历史）">${iconHTML("trash")}</button>
   </div>`).join(""):'<div class="tree-hint">没有符合条件的材料</div>';
 $("materials-list").querySelectorAll("[data-material]").forEach(b=>{
   b.onclick=()=>selectMaterial(b.dataset.material);
   b.ondblclick=()=>{selectMaterial(b.dataset.material);activateView("structure")};
 });
 $("materials-list").querySelectorAll("[data-remove-material]").forEach(b=>{
   b.onclick=()=>removeMaterial(b.dataset.removeMaterial);
 });
}
async function removeMaterial(mid){
 const m=S.materials.find(x=>x.id===mid);
 if(!m)return;
 if(!confirm(`确定从“已导入结构”中删除「${m.label}」吗？\\n原始计算源码和 examples 样例不会修改，已有任务和结果将保留。`))return;
 try{
   const result=await send(`/api/materials/${mid}/remove`,{confirm:true});
   if(S.selectedMaterial===mid){
     S.selectedMaterial=null;S.preview=null;
     clearStructureSelection();
   }
   await refreshMaterials();
   toast(result.archived?"已从工程树移除，历史计算及快照已保留":"结构已从工作区删除","success");
 }catch(error){toast("删除结构失败："+error.message,"error")}
}
function clearStructureSelection(){
 $("inspect-name").textContent="无选择";
 $("inspect-id").textContent="请选择左侧材料";
 for(const id of ["prop-formula","prop-atoms","prop-lengths","prop-angles","prop-pbc"])$(id).textContent="—";
 $("breadcrumb").textContent="当前工程 / 未选择结构";
 $("workspace-tag").textContent="NO DOCUMENT";
 $("status-material").textContent="未选择材料";
 $("viewer-empty").classList.remove("hidden");
 $("viewer-overlay").textContent="未载入结构";
 renderAtoms();requestAnimationFrame(drawStructure);
 setBackendUI();
}
function renderSamples(){
 $("samples-list").innerHTML=S.samples.length?S.samples.map(m=>`
   <button class="tree-item" data-sample="${esc(m.filename)}" title="点击从甲方 examples 复制该样例到工作区">
     <span class="materialicon">${iconHTML(fileIconFor(m.filename))}</span><span class="tree-name">${esc(m.filename)}</span><span class="tree-count">${formatSize(m.bytes)}</span>
   </button>`).join(""):'<div class="tree-hint">无法读取甲方的 examples/ 目录</div>';
 $("samples-list").querySelectorAll("[data-sample]").forEach(b=>b.onclick=()=>importSample(b.dataset.sample));
}
async function refreshMaterials(){
 S.materials=await api("/api/materials");S.samples=await api("/api/samples");
 renderSamples();renderMaterials();
 if(S.selectedMaterial&&!S.materials.some(m=>m.id===S.selectedMaterial))S.selectedMaterial=null;
 if(!S.selectedMaterial&&S.materials.length)selectMaterial(S.materials[0].id);
 else if(S.selectedMaterial)selectMaterial(S.selectedMaterial,true);
}
function selectMaterial(id,quiet=false){
 const m=S.materials.find(x=>x.id===id);if(!m)return;
 S.selectedMaterial=id;S.preview=m.preview;
 $("inspect-name").textContent=m.label;$("inspect-id").textContent=m.id.slice(0,12)+" · "+m.origin;
 $("prop-formula").textContent=m.formula;
 $("prop-atoms").textContent=m.atom_count;
 $("prop-lengths").textContent=m.preview.lengths.map(v=>fmt(v,3)).join(" / ");
 $("prop-angles").textContent=m.preview.angles.map(v=>fmt(v,1)).join(" / ");
 $("prop-pbc").textContent=m.preview.pbc.every(Boolean)?"是 (3D)":"否";
 $("breadcrumb").textContent="项目 / 材料库 / "+m.label;
 $("workspace-tag").textContent=m.preview.formula+" / "+m.atom_count+" atoms";
 $("status-material").textContent=m.formula+"  ·  "+m.label;
 $("viewer-overlay").textContent="CELL: "+m.formula+"   |   "+m.atom_count+" atoms"+(m.preview.preview_truncated?"  |  PREVIEW LIMITED":"");
 $("viewer-empty").classList.add("hidden");
 setBackendUI();
 if(!quiet){S.angleX=-.47;S.angleZ=.64;S.zoom=1}
 renderAtoms();renderMaterials();requestAnimationFrame(drawStructure);
 if(!quiet)setStatus("已选择材料："+m.label);
}
function renderAtoms(){
 const p=S.preview;if(!p){$("atoms-body").innerHTML='<tr><td colspan="5" class="empty-cell">暂无原子数据</td></tr>';return}
 $("atoms-body").innerHTML=p.symbols.map((sym,i)=>`<tr><td class="num">${i+1}</td><td>${esc(sym)}</td><td class="num">${fmt(p.positions[i][0],6)}</td><td class="num">${fmt(p.positions[i][1],6)}</td><td class="num">${fmt(p.positions[i][2],6)}</td></tr>`).join("");
 $("atom-footer").textContent=`共 ${p.atom_count} 个原子 · 当前显示 ${p.symbols.length} 行；原子坐标为原始笛卡尔坐标 (Å)${p.preview_truncated?"（预览上限 450）":""}`;
}
async function importSample(filename){
 try{
   const m=await send("/api/materials/from-sample",{filename});
   S.materials=await api("/api/materials");selectMaterial(m.id);activateView("structure");toast("已导入甲方示例："+filename,"success")
 }catch(e){toast("导入样例失败："+e.message,"error")}
}
async function uploadFile(file){
 if(!file)return;
 try{
   if(file.size>12*1024*1024)throw new Error("文件超过 12MB 限制");
   const m=await api("/api/materials/upload",{method:"POST",
     headers:{"X-Filename":encodeURIComponent(file.name),"Content-Type":"application/octet-stream"},
     body:await file.arrayBuffer()});
   S.materials=await api("/api/materials");selectMaterial(m.id);activateView("structure");
   toast("导入成功："+m.formula+" · "+m.atom_count+" 个原子","success")
 }catch(e){toast("晶体导入失败："+e.message,"error")}
}

// GPU evidence panel is created programmatically: existing index.html need not change.
function ensureGpuMonitor(){
  if($("bgp-gpu-monitor"))return;
  const anchor=$("backend-indicator")||$("preset-text");
  if(!anchor)return;
  const style=document.createElement("style");
  style.textContent=`
    #bgp-gpu-monitor{box-sizing:border-box;display:block;margin:9px 0;padding:11px;
      border:1px solid #aebfd0;border-radius:8px;background:#f5f9fc;color:#20364b;
      font:12px/1.5 "Segoe UI","Microsoft YaHei",sans-serif}
    #bgp-gpu-monitor[hidden]{display:none!important}
    #bgp-gpu-monitor .gpu-panel-title{display:flex;justify-content:space-between;
      gap:8px;align-items:center;margin-bottom:8px;font-weight:700}
    #bgp-gpu-monitor .gpu-panel-state{font-size:11px;color:#48617a;font-weight:500;
      background:#e5edf5;border-radius:20px;padding:2px 8px}
    #bgp-gpu-monitor .gpu-panel-grid{display:grid;grid-template-columns:minmax(78px,auto) minmax(0,1fr);
      gap:4px 8px}
    #bgp-gpu-monitor .gpu-panel-grid span{color:#596d80}
    #bgp-gpu-monitor .gpu-panel-grid strong{font-weight:600;min-width:0;overflow-wrap:anywhere}
    #bgp-gpu-monitor .gpu-panel-note{font-size:11px;color:#5c7084;
      border-top:1px solid #d6e0ea;margin-top:8px;padding-top:6px}
    #bgp-gpu-monitor a{display:inline-block;color:#185a88;font-size:12px;margin-top:7px}
    #bgp-gpu-monitor a[hidden]{display:none!important}
  `;
  document.head.append(style);
  const panel=document.createElement("section");
  panel.id="bgp-gpu-monitor";
  panel.setAttribute("aria-label","GPU任务运行与证据监控");
  panel.innerHTML=`
    <div class="gpu-panel-title">
      <span>GPU运行监控 · 实际采样</span>
      <span class="gpu-panel-state" id="gpu-panel-state">等待GPU任务</span>
    </div>
    <div class="gpu-panel-grid">
      <span>关联任务</span><strong id="gpu-job-id">—</strong>
      <span>计算方法</span><strong id="gpu-job-backend">—</strong>
      <span>GPU型号</span><strong id="gpu-device-name">—</strong>
      <span>GPU利用率</span><strong id="gpu-util-pct">—</strong>
      <span>显存占用</span><strong id="gpu-memory-usage">—</strong>
      <span>ABACUS进程</span><strong id="gpu-abacus-pids">—</strong>
      <span>进程级证据</span><strong id="gpu-evidence">未核验</strong>
      <span>采样时间(UTC)</span><strong id="gpu-sample-time">—</strong>
    </div>
    <a id="gpu-csv-download" hidden>下载当前任务GPU监控CSV</a>
    <div class="gpu-panel-note">
      GPU利用率和显存为显卡整卡数据；仅在NVIDIA进程列表中匹配ABACUS PID时，
      才显示“已确认进程级GPU关联”。运行任务的CSV会自动保存在任务目录。
    </div>`;
  anchor.insertAdjacentElement("afterend",panel);
  updateGpuMonitorVisibility();
}

function updateGpuMonitorVisibility(){
  const panel=$("bgp-gpu-monitor");
  if(!panel)return;
  const anyActive=S.jobs.some(j=>j.backend!=="gnn-hse06"&&["running","stopping","queued"].includes(j.state));
  const picked=selectedJob();
  panel.hidden=chosenBackend()==="gnn-hse06"&&!anyActive&&picked?.backend!=="abacus-hse06"&&picked?.backend!=="abacus-pbe0"&&picked?.backend!=="cider-pbe0";
}

function setGpuText(id,value){
  const node=$(id);
  if(node)node.textContent=value;
}

async function refreshGpuMonitor(){
  const panel=$("bgp-gpu-monitor");
  if(!panel||S.gpuRequestBusy)return;
  updateGpuMonitorVisibility();
  if(panel.hidden)return;
  const selected=selectedJob();
  const active=S.jobs.find(j=>j.backend!=="gnn-hse06"&&j.state==="running");
  const job=(selected&&selected.backend!=="gnn-hse06")?selected:active;
  const link=$("gpu-csv-download");
  if(!job){
    setGpuText("gpu-panel-state","等待GPU任务");
    setGpuText("gpu-job-id","未提交GPU计算");
    for(const id of ["gpu-job-backend","gpu-device-name","gpu-util-pct",
      "gpu-memory-usage","gpu-abacus-pids","gpu-sample-time"])setGpuText(id,"—");
    setGpuText("gpu-evidence","未核验");
    link.hidden=true;
    return;
  }
  S.gpuRequestBusy=true;
  try{
    const data=await api(`/api/jobs/${encodeURIComponent(job.id)}/gpu`);
    if(!S.jobs.some(j=>j.id===job.id))return;
    const sample=data.sample;
    setGpuText("gpu-panel-state",tr(data.state)+(data.state==="running"?" · 实时采样":" · 历史记录"));
    setGpuText("gpu-job-id",job.id);
    setGpuText("gpu-job-backend",data.backend);
    setGpuText("gpu-sample-time",sample?.timestamp_utc||"尚未采样");
    if(!sample){
      setGpuText("gpu-device-name","等待GPU采样");
      setGpuText("gpu-util-pct","—");
      setGpuText("gpu-memory-usage","—");
      setGpuText("gpu-abacus-pids","等待进程");
      setGpuText("gpu-evidence","尚无可用证据");
    }else{
      setGpuText("gpu-device-name",sample.available?(sample.name||"设备名称未知"):"监控不可用");
      const u=sample.utilization_percent;
      setGpuText("gpu-util-pct",u===null||u===undefined||u===""?"不可用":`${u}%（整卡）`);
      const used=sample.memory_used_mb,total=sample.memory_total_mb;
      setGpuText("gpu-memory-usage",used==null||total==null?"不可用":`${used} / ${total} MiB（整卡）`);
      const pids=Array.isArray(sample.abacus_pids)?sample.abacus_pids:[];
      setGpuText("gpu-abacus-pids",pids.length?pids.join(", "):"当前采样未识别");
      setGpuText("gpu-evidence",sample.process_gpu_match?
        "已确认：ABACUS PID出现在NVIDIA计算进程列表":
        pids.length?"已识别ABACUS子进程；未获得进程级GPU匹配":
        "未识别ABACUS进程级GPU关联");
      if(!sample.available&&sample.error){
        setGpuText("gpu-evidence","监控数据不可用："+sample.error);
      }
    }
    link.hidden=!data.monitor_csv_exists;
    if(data.monitor_csv_exists){
      link.href=`/api/jobs/${encodeURIComponent(job.id)}/download?path=gpu_monitor.csv`;
      link.setAttribute("download",`gpu_monitor_${job.id}.csv`);
    }
  }catch(err){
    setGpuText("gpu-panel-state","监控接口异常");
    setGpuText("gpu-evidence",err.message);
    link.hidden=true;
  }finally{
    S.gpuRequestBusy=false;
  }
}

function chosenBackend(){return $("backend-select").value}
function updateCalculationCategory(){
 const cat=$("compute-category").value;
 const methods=computeGroups[cat]?.methods||computeGroups.ml.methods;
 const selected=$("backend-select").value;
 $("backend-select").innerHTML=methods.map(([id,label])=>`<option value="${id}">${esc(label)}</option>`).join("");
 if(methods.some(row=>row[0]===selected))$("backend-select").value=selected;
 setBackendUI();
}
function setBackendUI(){
  const m=chosenBackend(),ready=S.environment?.methods?.find(x=>x.id===m);
  const needGPU=m!=="gnn-hse06";
  const abacus=$("abacus-path").value.trim();
  const configured=S.environment?.abacus;
  const defaultPath=!!configured&&abacus===configured.path;
  const validDefault=defaultPath&&configured.executable;
  const customPath=abacus.startsWith("/")&&!defaultPath;
  const canSubmitGpu=validDefault||customPath;
  const indicator=$("backend-indicator");
  indicator.className="backend-indicator "+(needGPU?(validDefault?"good":"bad"):(ready?.ready?"good":"bad"));
  if(needGPU){
    if(validDefault){
      indicator.textContent="✓ 已识别本机 ABACUS GPU 程序；提交前建议运行 doctor";
    }else if(customPath){
      indicator.textContent="⚠ 自定义 ABACUS 路径将在提交时由服务端验证，请运行 doctor";
    }else if(defaultPath){
      indicator.textContent="⚠ 未找到已配置的 ABACUS 可执行程序，请检查安装路径";
    }else{
      indicator.textContent="⚠ 请填写 ABACUS 可执行文件的 Linux 绝对路径";
    }
  }else{
    indicator.textContent=ready?(ready.ready?"✓ GNN 后端环境可用":"⚠ GNN 后端尚不可用："+ready.reason):"GNN 后端状态尚未检查";
  }
  $("device-select").disabled=needGPU;
  $("abacus-path").disabled=!needGPU;
  $("device-description").textContent=needGPU?"此方法使用已编译的 GPU ABACUS；后台自动传入 --abacus 和 Linux 工作目录。":"GNN 模型预测不需要 ABACUS；可选择 CPU 或 CUDA。";
  $("preset-text").textContent=needGPU?(m==="cider-pbe0"?"CIDER-PBE0 使用数据驱动非局域泛函，锁定预设 v1.2，xmix=0.15；建议运行 ABACUS doctor 检查。":"GPU 带隙计算调用 ABACUS HSE06 或 PBE0。科学参数由引擎锁定预设控制。"):"机器学习带隙误差修正调用 gnn-hse06（MACE 模型与冻结分类头），非第一性原理 GPU 计算。";
  $("tool-run").disabled=!S.selectedMaterial||!S.environment?.installed||S.environment?.source_match===false||
    (needGPU?!canSubmitGpu:ready?.ready===false);
  $("inspector-run").disabled=$("tool-run").disabled;
  updateGpuMonitorVisibility();
}
async function refreshEnvironment(){
 const v=await api("/api/system");S.environment=v;
 // Keep manually entered paths, initialize default GPU ABACUS from server.
 if(!S.abacusEdited&&v.abacus?.path)$("abacus-path").value=v.abacus.path;
 $("title-health").textContent=!v.installed?"bgp 未就绪":v.source_match===false?"注意：计算源码位置不匹配":"bgp 已安装";
 $("health-dot").className="health-dot "+(v.installed&&v.source_match!==false?"good":"bad");
 $("toolbar-led").className="toolbar-led "+(v.installed&&v.source_match!==false?"good":"bad");
 $("current-source").textContent="ENGINE "+v.source;
 $("work-root-label").textContent=v.data;
 $("status-engine").textContent=v.version||"bgp 未安装";
 setBackendUI();
}
async function submitJob(){
 const material=selectedMaterial();if(!material){toast("请先从工程树选择一个晶体文件","error");return}
 if(!S.environment?.installed){toast("未找到 bgp，请先检查计算环境","error");return}
 const m=chosenBackend(),ready=S.environment?.methods?.find(x=>x.id===m);
 if(m==="gnn-hse06"&&ready&&!ready.ready){toast("GNN 后端当前不可用："+ready.reason,"error");return}
 if(m!=="gnn-hse06"&&!$("abacus-path").value.trim().startsWith("/")){
   toast("请填写 ABACUS 的 Linux 绝对路径，或重新检查环境","error");return;
 }
 try{
   const j=await send("/api/jobs",{
     material_id:material.id,backend:m,device:$("device-select").value,
     abacus:$("abacus-path").value.trim(),note:$("job-note").value.trim()
   });
   S.selectedJob=j.id;activateDock("output");
   $("console").textContent="[GUI] 已提交任务 "+j.id+"\n[GUI] 等待计算队列调度...\n";
   toast("任务已提交："+j.id.slice(0,8),"success");
   setStatus("已提交 "+m+" 计算任务");
   await refreshJobs(true);
 }catch(e){toast("任务提交失败："+e.message,"error")}
}
async function stopJob(){
 const j=selectedJob();if(!j)return;
 if(!["queued","running","stopping"].includes(j.state)){toast("任务已结束，无法终止");return}
 try{await send(`/api/jobs/${j.id}/cancel`,{});toast("已发送中断请求");await refreshJobs(true)}
 catch(e){toast(e.message,"error")}
}
async function duplicateJob(){
 const j=selectedJob();if(!j)return;
 try{
   const next=await send(`/api/jobs/${j.id}/duplicate`,{});
   S.selectedJob=next.id;toast("已重新排队："+next.id.slice(0,8),"success");
   await refreshJobs(true)
 }catch(e){toast("重复运行失败："+e.message,"error")}
}
async function refreshJobs(force=false){
 const items=await api("/api/jobs");
 const previous=S.jobs.find(j=>j.id===S.selectedJob);
 S.jobs=items;
 $("job-tab-count").textContent=items.length;
 const active=items.filter(x=>["queued","running","stopping"].includes(x.state));
 $("queue-count").textContent=active.length;
 $("status-jobs").textContent=`${items.length} tasks · ${active.length} active`;
 $("tool-stop").disabled=!selectedJob()||!["queued","running","stopping"].includes(selectedJob().state);
 $("tool-retry").disabled=!selectedJob();
 renderJobsTable();renderQueue();renderCompare();
 if(!S.selectedJob&&items.length)S.selectedJob=items[0].id;
 const current=selectedJob();
 if(current){updateJobProps(current);
   if(!previous||force||previous.state!==current.state||previous.result_path!==current.result_path){renderResult();renderReport();renderProblems()}
   if(S.view==="artifacts"&&current.result_path&&!previous?.result_path)loadArtifacts();
   if(force||["running","stopping","queued"].includes(current.state)||S.logSize<0)await refreshLog(current);
 }
}
function selectJob(id,show=false){
 const j=S.jobs.find(r=>r.id===id);if(!j)return;
 S.selectedJob=id;S.logSize=-1;S.clearedAt=0;
 const m=S.materials.find(x=>x.id===j.material_id);
 if(m)selectMaterial(m.id,true);
 updateJobProps(j);renderJobsTable();renderResult();renderReport();renderProblems();refreshLog(j);
 if(show)activateView("results");
 setStatus("选中任务 "+j.id.slice(0,8)+" · "+tr(j.state));
 void refreshGpuMonitor();
}
function updateJobProps(j){
 $("prop-job-id").textContent=j.id.slice(0,12);
 $("prop-job-state").textContent=tr(j.state);
 $("prop-job-backend").textContent=j.backend;
 $("prop-job-gap").textContent=fmt(gapValue(j),7);
 $("prop-job-dir").textContent=j.run_dir||"尚未生成";
 $("tool-stop").disabled=!["queued","running","stopping"].includes(j.state);
 $("tool-retry").disabled=false;
 $("tool-report").disabled=!j;
}
function renderJobsTable(){
 const text=$("jobs-search").value.trim().toLowerCase(),status=$("jobs-filter").value;
 const filtered=S.jobs.filter(j=>(status==="all"||j.state===status)&&(j.material_label+" "+j.material_formula+" "+j.backend+" "+j.id).toLowerCase().includes(text));
 $("jobs-summary").textContent=`显示 ${filtered.length} / 总计 ${S.jobs.length} 条记录`;
 $("jobs-body").innerHTML=filtered.length?filtered.map(j=>{
   const gap=gapValue(j), type=jobResult(j)?.prediction?.gap_type;
   return `<tr data-job="${esc(j.id)}" class="${S.selectedJob===j.id?"row-selected":""}">
    <td>${esc(j.id.slice(0,11))}</td><td title="${esc(j.material_label)}">${esc(j.material_formula)} · ${esc(j.material_label)}</td>
    <td>${esc(names[j.backend]||j.backend)}</td><td><span class="status-pill ${esc(j.state)}">${esc(tr(j.state))}</span></td>
    <td>${gap===null?"—":fmt(gap,7)}</td><td>${esc(type?tr(type):"未提供")}</td>
    <td>${fmt(jobResult(j)?.elapsed_seconds,1)}</td><td>${esc(dt(j.submitted_at))}</td></tr>`
 }).join(""):'<tr><td colspan="8" class="empty-cell">当前无符合筛选的任务</td></tr>';
 $("jobs-body").querySelectorAll("[data-job]").forEach(tr=>{
   tr.onclick=()=>selectJob(tr.dataset.job);
   tr.ondblclick=()=>selectJob(tr.dataset.job,true);
 });
}
function renderQueue(){
 const active=S.jobs.filter(j=>["queued","running","stopping"].includes(j.state));
 $("queue-list").innerHTML=active.length?active.map(j=>`<div class="queue-entry" data-id="${esc(j.id)}">
   <span class="status-pill ${esc(j.state)}">${esc(tr(j.state))}</span><span>${esc(j.material_formula)}</span>
   <span>${esc(j.backend)}</span><span class="spacer"></span><span>${esc(j.id.slice(0,10))}</span>
   </div>`).join(""):"当前没有排队或执行中的任务。";
 $("queue-list").querySelectorAll("[data-id]").forEach(e=>e.onclick=()=>selectJob(e.dataset.id))
}
async function refreshLog(j){
 if(!j)return;
 try{
   const r=await api(`/api/jobs/${j.id}/log`);
   if(S.selectedJob!==j.id)return;
   if(S.logSize!==r.size){
     S.logSize=r.size;
     S.currentLog=r.text;
     $("console").textContent=r.text||`[GUI] ${tr(j.state)} · 等待计算日志...`;
     if($("auto-scroll").checked)$("console").scrollTop=$("console").scrollHeight;
   }
 }catch(e){console.warn("log:",e)}
}

function reportRows(rows){
 return rows.map(([key,value])=>{
   let content=value;
   if(typeof content==="object"&&content!==null)content=JSON.stringify(content,null,2);
   if(content===null||content===undefined||content==="")content="未提供";
   return `<tr><th>${esc(key)}</th><td class="report-value">${esc(content)}</td></tr>`;
 }).join("");
}
function reportValue(v){
 if(v===null||v===undefined||v==="")return "未提供";
 return v;
}
function renderReport(){
 const job=selectedJob(), raw=job?.result;
 const display=!!job && (raw||["success","failed","interrupted"].includes(job.state));
 $("report-empty").classList.toggle("hidden",display);
 $("report-sheet-wrap").classList.toggle("hidden",!display);
 $("report-save").disabled=!display;
 $("report-print").disabled=!display;
 if(!display){
   $("report-empty").textContent=job?
     `任务 ${job.id.slice(0,10)} 正在${tr(job.state)}。待计算结束后在当前工作区显示报告。`:
     "请在任务管理中选择已结束任务，报告将在当前页面的本标签内显示。";
   return;
 }
 const p=raw?.prediction||{}, valid=jobResult(job)!==null;
 const value=valid?gapValue(job):null;
 const type=categoryOfBackend(job.backend);
 const cat=computeGroups[type]?.label||"未提供";
 const probability=typeof p.gap_type_probability==="number"?
   `${(100*p.gap_type_probability).toFixed(2)}%（直接/间接类型概率）`:"未提供";
 $("report-basic").innerHTML=reportRows([
   ["任务编号",job.id],
   ["材料名称",job.material_label],
   ["化学式",job.material_formula],
   ["计算类型",cat],
   ["计算后端",job.backend],
   ["计算方法",raw?.method],
   ["任务状态",tr(job.state)],
   ["创建时间",job.submitted_at],
   ["结束时间",job.finished_at]
 ]);
 $("report-prediction").innerHTML=reportRows([
   ["带隙 (eV)",value===null?"未提供（仅成功计算展示数值）":value.toFixed(7)],
   ["电子态分类",valid?tr(p.classification):"未提供"],
   ["直接 / 间接",valid&&p.gap_type?tr(p.gap_type):"未提供（不能据此推断）"],
   ["带隙类型概率",valid?probability:"未提供"],
   ["收敛信息",raw?.converged===true?"已收敛":raw?.converged===false?"未收敛":"不适用或未报告"],
   ["执行耗时",typeof raw?.elapsed_seconds==="number"?`${raw.elapsed_seconds.toFixed(3)} s`:"未提供"]
 ]);
 $("report-settings").innerHTML=reportRows([
   ["计算类型",cat],
   ["算法后端",job.backend],
   ["计算设备",job.backend==="gnn-hse06"?job.device:"ABACUS 运行配置"],
   ["ABACUS 路径",job.backend==="gnn-hse06"?"不适用":job.abacus_path||"未指定（由引擎自行解析）"],
   ["受控参数",raw?.settings],
   ["任务备注",job.note]
 ]);
 $("report-provenance").innerHTML=reportRows([
   ["运行 ID",raw?.run_id],
   ["模型 SHA-256",raw?.provenance?.model_sha256],
   ["输入文件 SHA-256",raw?.structure?.original_sha256||raw?.structure?.structure_sha256],
   ["模型与来源完整信息",raw?.provenance],
   ["引擎诊断",raw?.diagnostics],
   ["任务结果文件",job.result_path],
   ["计算输出目录",job.run_dir],
   ["退出码",job.exit_code]
 ]);
 let messages=[...(Array.isArray(raw?.warnings)?raw.warnings:[])];
 if(job.error)messages.unshift(job.error);
 if(raw?.converged===false)messages.unshift("收敛性不足：不应将该结果当作有效数值带隙。");
 if(!valid)messages.unshift("任务未取得可确认的成功结果，本报告不提供虚构带隙数值。");
 if(type==="ml")messages.push("GNN 模型属于 legacy/provisional；请保留模型哈希和科学资产验证记录。");
 $("report-warnings").innerHTML=messages.length?
   `<div class="report-warning-title">运行提示及科学警告</div>${messages.map(v=>`<p>${esc(v)}</p>`).join("")}`:
   `<div class="report-warning-title">运行提示及科学警告</div><p>结果文件未报告额外警告；这不代表已经通过验收。</p>`;
}
function reportText(){
 const job=selectedJob(),r=job?.result;
 if(!job)return "";
 const p=r?.prediction||{},valid=jobResult(job)!==null;
 const displayGap=valid&&typeof p.gap_ev==="number"?p.gap_ev.toFixed(7)+" eV":"未提供";
 return [
  "GNNBGP Studio / 材料带隙计算报告",
  "=".repeat(50),
  "任务编号："+job.id,
  "材料："+job.material_label+" / "+job.material_formula,
  "计算类型："+computeGroups[categoryOfBackend(job.backend)].label,
  "计算后端："+job.backend,
  "任务状态："+tr(job.state),
  "带隙："+displayGap,
  "电子态分类："+(valid?tr(p.classification):"未提供"),
  "直接/间接："+(valid&&p.gap_type?tr(p.gap_type):"未提供"),
  "类型概率："+(valid&&typeof p.gap_type_probability==="number"?(p.gap_type_probability*100).toFixed(2)+"%":"未提供"),
  "收敛："+(r?.converged===true?"是":r?.converged===false?"否":"未报告"),
  "方法："+reportValue(r?.method),
  "运行时间："+reportValue(r?.elapsed_seconds),
  "模型 SHA256："+reportValue(r?.provenance?.model_sha256),
  "结果文件："+reportValue(job.result_path),
  "输出目录："+reportValue(job.run_dir),
  "",
  "设置 / settings:",
  JSON.stringify(r?.settings||{},null,2),
  "",
  "模型来源 / provenance:",
  JSON.stringify(r?.provenance||{},null,2),
  "",
  "诊断 / diagnostics:",
  JSON.stringify(r?.diagnostics||{},null,2),
  "",
  "警告 / warnings:",
  [...(r?.warnings||[]),...(job.error?[job.error]:[])].join("\n")||"无记录",
  "",
  "说明：本报告来自当前任务 result.json，不作人工验收判定。"
 ].join("\n");
}
function saveReport(){
 if($("report-save").disabled)return;
 const job=selectedJob();
 const text=reportText();
 if(!text)return;
 const url=URL.createObjectURL(new Blob(["\ufeff"+text],{type:"text/plain;charset=utf-8"}));
 const a=document.createElement("a");
 a.href=url;a.download=`gnnbgp-report-${job.id.slice(0,10)}.txt`;
 a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);
 toast("已导出所选任务的文本报告","success");
}

function renderResult(){
 const j=selectedJob(),r=jobResult(j);
 $("result-raw").disabled=!j?.result_path;
 $("result-export").disabled=!r;
 $("tool-report").disabled=!j;
 if(!r){$("result-empty").classList.remove("hidden");$("result-content").classList.add("hidden");
   $("result-empty").textContent=j?`当前任务：${tr(j.state)}。${j.error||"尚无可验证的成功计算结果。"} 所有带隙数值仅在 bgp 成功返回结果后展示。`:"请选择任务以查看结果";return}
 $("result-empty").classList.add("hidden");$("result-content").classList.remove("hidden");
 const p=r.prediction||{}, v=gapValue(j);
 $("result-gap").textContent=v===null?"—":fmt(v,7);
 $("result-state").textContent=tr(r.status);
 $("result-class").textContent=tr(p.classification);
 $("result-type").textContent=p.gap_type?tr(p.gap_type):"未提供";
 $("result-prob").textContent=typeof p.gap_type_probability==="number"?(p.gap_type_probability*100).toFixed(2)+"%":"未提供";
 $("result-converged").textContent=r.converged===true?"已收敛":r.converged===false?"未收敛":"不适用 / 未报告";
 $("result-runtime").textContent=fmt(r.elapsed_seconds,3)+" s";
 const meta=[["Run ID",r.run_id],["Method",r.method],["Backend",r.backend],
   ["Material",r.structure?.formula],["Original SHA256",r.structure?.original_sha256],
   ["Structure SHA256",r.structure?.structure_sha256],["Artifact directory",r.artifact_path],
   ["Started at",r.started_at],["Finished at",r.finished_at]];
 $("result-meta").innerHTML=meta.map(([k,v])=>`<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join("");
 const warnings=[...(r.warnings||[])];
 if(r.prediction?.classification==="partial_occupancy")warnings.push("部分占据态：需要结合电子占据证据分析 0 eV 的物理含义。");
 if(warnings.length){$("result-warning").classList.remove("hidden");$("result-warning").textContent=warnings.join("\n")}
 else $("result-warning").classList.add("hidden");
 renderDetail();
}
function renderDetail(){
 const r=jobResult(selectedJob());const data=r?.[S.detail]||{};
 $("result-details").textContent=JSON.stringify(data,null,2);
 document.querySelectorAll(".detail-tab").forEach(e=>e.classList.toggle("active",e.dataset.detail===S.detail));
}
function renderProblems(){
 const j=selectedJob(),r=j?.result;const problems=[];
 if(j?.error)problems.push({type:"错误",text:j.error});
 for(const w of r?.warnings||[])problems.push({type:"警告",text:w});
 if(r?.converged===false)problems.push({type:"未收敛",text:"甲方结果显示 SCF 未收敛，结果不得按成功带隙展示。"});
 $("problems-count").textContent=problems.length;
 $("problems-list").innerHTML=problems.length?problems.map(p=>`<div class="problem-entry"><b>[${esc(p.type)}]</b> ${esc(p.text)}</div>`).join(""):"当前选中任务没有记录到问题。";
}
function renderCompare(){
 const valid=S.jobs.filter(j=>gapValue(j)!==null).slice(0,30);
 const scale=Math.max(6, ...valid.map(j=>gapValue(j)));
 $("compare-chart").innerHTML=valid.length?valid.map(j=>`<div class="chart-row">
 <div class="chart-label" title="${esc(j.material_label+" · "+j.backend)}">${esc(j.material_formula)} · ${esc(j.backend)}</div>
 <div class="chart-track"><div class="chart-fill" data-percentage="${Math.min(100,Math.max(0,gapValue(j)/scale*100)).toFixed(3)}"></div></div>
 <div class="chart-value">${fmt(gapValue(j),5)} eV</div></div>`).join(""):'<div class="context-empty">暂无有效的已完成带隙计算</div>';
 $("compare-chart").querySelectorAll("[data-percentage]").forEach(el=>{
  el.style.width=el.dataset.percentage+"%";
 });
}
async function loadArtifacts(){
 const j=selectedJob();if(!j){$("artifacts-body").innerHTML='<tr><td colspan="3" class="empty-cell">请选择一个任务</td></tr>';return}
 $("artifacts-dir").textContent=j.run_dir||"任务 "+j.id+" 尚未产生计算目录";
 try{
   const files=await api(`/api/jobs/${j.id}/files`);
   $("artifacts-body").innerHTML=files.length?files.map(f=>{
    const link=`/api/jobs/${j.id}/download?path=${encodeURIComponent(f.path)}`;
    return `<tr><td><span class="artifact-name">${iconHTML(fileIconFor(f.path),"ui-icon artifact-file-icon")}${esc(f.path)}</span></td><td>${formatSize(f.bytes)}</td><td><a class="filelink" href="${esc(link)}" download>${iconHTML("download")}下载文件</a></td></tr>`
   }).join(""):'<tr><td colspan="3" class="empty-cell">没有找到任务产物文件</td></tr>';
 }catch(e){toast("读取任务产物失败："+e.message,"error")}
}
async function diagnostic(action){
 showModal("环境诊断 — "+({assets:"模型资产校验",doctor:"ABACUS 运行检查",backends:"计算后端状态"}[action]||action),"<p>正在运行甲方 CLI 诊断命令，请勿关闭计算服务...</p>");
 try{const r=await send("/api/diagnostic",{action,abacus:$("abacus-path").value.trim()});
   $("modal-body").innerHTML=`<p><b>${r.ok?"✓ 检查通过":"⚠ 检查未通过"}</b> · 退出码 ${r.code}</p><pre>${esc(r.output)}</pre>`;
 }catch(e){$("modal-body").textContent="诊断失败："+e.message}
}
function showModal(title,html){
 $("modal-title").textContent=title;$("modal-body").innerHTML=html;$("modal-backdrop").classList.remove("hidden")
}
function closeModal(){$("modal-backdrop").classList.add("hidden")}
function showEnvironment(){
 const e=S.environment;
 const table=(e?.methods||[]).map(m=>`<tr><th>${esc(m.id)}</th><td>${esc(m.reason)}</td></tr>`).join("");
 showModal("系统环境及计算能力",`
   <p><b>已安装引擎：</b> ${esc(e?.version||"—")}</p>
   <p><b>源码匹配状态：</b> ${e?.source_match===true?"✓ 当前虚拟环境加载了指定的新源码":e?.source_match===false?"⚠ 当前虚拟环境可能安装的是其他源码路径":"尚未验证"}</p>
   <table class="kv-table"><tbody>
     <tr><th>源码目录</th><td>${esc(e?.source)}</td></tr>
     <tr><th>bgp CLI</th><td>${esc(e?.bgp)}</td></tr>
     <tr><th>Python 模块位置</th><td>${esc(e?.engine_module||"未检查")}</td></tr>
     <tr><th>Python 解释器</th><td>${esc(e?.python)}</td></tr>
     <tr><th>工作区</th><td>${esc(e?.data)}</td></tr>${table}
   </tbody></table>
   <p>诊断操作执行甲方原生 bgp 命令，不会重装/更新计算依赖。</p>
   <div class="modal-choice">
    <button data-diag="backends">${iconHTML("list-todo")}重新检测四种后端 <small>bgp backends</small></button>
    <button data-diag="assets">${iconHTML("shield-check")}验证科学资产哈希 <small>bgp assets verify</small></button>
    <button data-diag="doctor">${iconHTML("cpu")}检查 ABACUS 可执行程序 <small>bgp doctor</small></button>
   </div>`);
 $("modal-body").querySelectorAll("[data-diag]").forEach(b=>b.onclick=()=>diagnostic(b.dataset.diag));
}
function showHelp(){
 showModal("GNNBGP 工业工作台 · 使用帮助",`
 <p><b>软件适配：</b>甲方新源码 <code>/mnt/d/program/gnnbgp</code>，通过 <code>bgp predict</code> 命令运行真实计算，并读取 <code>result.json</code>。</p>
 <p><b>基本操作：</b>① 从左侧导入 CIF/POSCAR 或选用 Si/InSb 等示例；② 右侧选择计算后端；③ 点击「启动计算」；④ 在底部查看日志、任务状态；⑤ 在「计算结果」中查看 eV、类型、溯源与科学参数。</p>
 <p><b>快捷键：</b>Ctrl+O 导入结构；Ctrl+Enter 启动计算；F5 刷新；Esc 关闭对话框。</p>
 <p><b>注意：</b>GNN 仅用于预估带隙及其直接/间接类型；模型为 legacy/provisional。ABACUS/CIDER 需要先单独编译并验证可执行文件。当前源码无可调用的真实 E–k 能带计算命令，所以不显示虚构能带。</p>
 <p><b>安全性：</b>软件只允许 127.0.0.1 本地访问；不要用它作为公网/多用户服务。</p>`);
}
function showAbout(){
 showModal("关于 GNNBGP Industrial Workbench",`
   <p><b>GNNBGP Studio · Industrial UI v0.6</b></p>
   <p>这是围绕甲方新 <code>gnnbgp</code> 源码的独立工业工作台，非计算模型重写。所有材料结构均由 ASE 读取并保留源文件，所有数值结果均来自真实 bgp 计算产生的 JSON 文件。</p>
   <p>引擎：<code>bandgap-predictor</code> 0.1.0 · 默认工作环境：<code>~/BandGap-Studio/.venv</code></p>
   <p>可复现性：保留每次计算独立输出、原始输入、日志、模型 SHA256、科学参数与来源信息。</p>`);
}
function exportJobsCSV(){
 exportCSV("gnnbgp-jobs.csv",[
  ["job_id","material","formula","backend","state","gap_ev","gap_type","elapsed_seconds","submitted_utc","run_dir"],
  ...S.jobs.map(j=>[j.id,j.material_label,j.material_formula,j.backend,j.state,
    gapValue(j)??"",jobResult(j)?.prediction?.gap_type||"",jobResult(j)?.elapsed_seconds??"",j.submitted_at,j.run_dir])
 ]);toast("任务清单 CSV 已导出","success")
}
function exportResultCSV(){
 const j=selectedJob(),r=jobResult(j);if(!r)return;
 exportCSV("gnnbgp-result-"+j.id.slice(0,8)+".csv",[
   ["field","value"],["run_id",r.run_id],["material",j.material_label],["backend",r.backend],
   ["method",r.method],["gap_ev",r.prediction?.gap_ev],["classification",r.prediction?.classification],
   ["gap_type",r.prediction?.gap_type],["gap_type_probability",r.prediction?.gap_type_probability],
   ["converged",r.converged],["elapsed_seconds",r.elapsed_seconds],
   ["model_sha256",r.provenance?.model_sha256],["artifact_path",r.artifact_path]
 ]);toast("结果摘要 CSV 已导出","success")
}
function exportCompareCSV(){
 exportCSV("gnnbgp-gap-comparison.csv",[
   ["formula","material","backend","gap_ev","gap_type","method"],
   ...S.jobs.filter(j=>gapValue(j)!==null).map(j=>[
     j.material_formula,j.material_label,j.backend,gapValue(j),jobResult(j)?.prediction?.gap_type||"",jobResult(j)?.method
   ])
 ]);toast("比较数据 CSV 已导出","success")
}
function rawResult(){
 const j=selectedJob();if(!j?.result_path)return;
 showModal("原始 result.json — "+j.id.slice(0,8),`<pre>${esc(JSON.stringify(j.result,null,2))}</pre>`);
}
function report(){
 if(!selectedJob()){toast("请先在任务管理中选择任务","error");return}
 activateView("report");
}
function menuItems(which){
 const menus={
 file:[["导入晶体结构…","Ctrl+O",()=>chosenFile()],["载入测试示例…","",openSampleDialog],["管理当前材料…","",manageMaterial],["—"],["导出当前结果 CSV","",exportResultCSV],["导出任务清单 CSV","",exportJobsCSV]],
 edit:[["管理当前材料…","",manageMaterial],["—"],["重复执行选中任务","",duplicateJob],["终止选中任务","",stopJob],["—"],["清空日志显示（非删除日志）","",()=>{$("console").textContent="";S.logSize=-1}]],
 view:[["结构视图","",()=>activateView("structure")],["计算结果","",()=>activateView("results")],["计算报告","",()=>activateView("report")],["任务管理","",()=>activateView("jobs")],["数据比较","",()=>activateView("compare")],["输出文件","",()=>activateView("artifacts")]],
 tools:[["启动计算","Ctrl+Enter",submitJob],["刷新状态","F5",refreshAll],["—"],["系统环境检查…","",showEnvironment],["验证模型资产…","",()=>diagnostic("assets")],["ABACUS 诊断…","",()=>diagnostic("doctor")]],
 help:[["使用帮助","",showHelp],["关于软件","",showAbout]]
 };
 return menus[which]||[];
}
const menuIconNames={
 "导入晶体结构…":"file-up",
 "载入测试示例…":"folder-open",
 "管理当前材料…":"edit",
 "导出当前结果 CSV":"download",
 "导出任务清单 CSV":"table",
 "重复执行选中任务":"rotate-cw",
 "终止选中任务":"stop",
 "清空日志显示（非删除日志）":"eraser",
 "结构视图":"atom",
 "计算结果":"file-text",
 "计算报告":"report",
 "任务管理":"list-todo",
 "数据比较":"chart-column",
 "输出文件":"files",
 "启动计算":"play",
 "刷新状态":"refresh-cw",
 "系统环境检查…":"settings",
 "验证模型资产…":"shield-check",
 "ABACUS 诊断…":"cpu",
 "使用帮助":"help",
 "关于软件":"info"
};
function toggleMenu(which){
 const overlay=$("menu-overlay");
 if(S.menu===which&&!overlay.classList.contains("hidden")){hideMenus();return}
 const trigger=document.querySelector(`[data-menu="${which}"]`);
 overlay.style.left=trigger.offsetLeft+"px";
 $("menu-items").innerHTML=menuItems(which).map((i,n)=>i[0]==="—"?'<div class="menu-line"></div>':
   `<button class="menu-item" data-menu-index="${n}"><span class="menu-item-leading">${iconHTML(menuIconNames[i[0]]||"file-text")}<span>${esc(i[0])}</span></span><kbd>${esc(i[1])}</kbd></button>`).join("");
 $("menu-items").querySelectorAll("[data-menu-index]").forEach(b=>b.onclick=()=>{
   const fn=menuItems(which)[Number(b.dataset.menuIndex)][2];hideMenus();fn?.()
 });
 document.querySelectorAll(".menu-trigger").forEach(x=>x.classList.toggle("menu-open",x.dataset.menu===which));
 overlay.classList.remove("hidden");S.menu=which
}
function hideMenus(){$("menu-overlay").classList.add("hidden");S.menu=null;document.querySelectorAll(".menu-trigger").forEach(x=>x.classList.remove("menu-open"))}
function chosenFile(){$("file-input").value="";$("file-input").click()}
function manageMaterial(){
 const m=selectedMaterial();if(!m){toast("先从左侧选择材料","error");return}
 showModal("管理工作区材料",`<p><b>${esc(m.label)}</b> · ${esc(m.formula)} · ${m.atom_count} atoms</p>
   <p>这里仅管理工作区中的复制文件，不更改甲方源代码和 examples 目录。</p>
   <div class="modal-choice">
    <button id="material-rename">${iconHTML("edit")}重命名显示名称 <small>不修改晶体文件内容</small></button>
    <button id="material-remove">${iconHTML("trash")}删除结构 <small>保留已有计算任务和结果</small></button>
   </div>`);
 $("material-rename").onclick=async()=>{
   const value=prompt("新的材料显示名称：",m.label);if(value===null)return;
   try{await send(`/api/materials/${m.id}/rename`,{label:value});
     await refreshMaterials();selectMaterial(m.id,true);closeModal();toast("已重命名","success")}
   catch(e){toast(e.message,"error")}
 };
 $("material-remove").onclick=async()=>{
   closeModal();await removeMaterial(m.id);
 };
}
function openSampleDialog(){
 showModal("载入甲方测试晶体",`<p>从 <code>${esc(S.environment?.source||"/mnt/d/program/gnnbgp")}/examples</code> 复制到本地项目库，不修改甲方原始文件。</p>
  <div class="modal-choice">${S.samples.map(s=>`<button data-sample-pick="${esc(s.filename)}">${iconHTML("file-plus")}${esc(s.filename)} <small>${formatSize(s.bytes)}</small></button>`).join("")}</div>`);
 $("modal-body").querySelectorAll("[data-sample-pick]").forEach(b=>b.onclick=()=>{closeModal();importSample(b.dataset.samplePick)})
}
function handleKeys(e){
 if(e.key==="Escape"){closeModal();hideMenus();return}
 const target=e.target;if(["INPUT","TEXTAREA","SELECT"].includes(target?.tagName))return;
 if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="o"){e.preventDefault();chosenFile()}
 if((e.ctrlKey||e.metaKey)&&e.key==="Enter"){e.preventDefault();submitJob()}
 if(e.key==="F5"){e.preventDefault();refreshAll()}
}
function makeSplitters(){
 const css=document.documentElement;
 [["drag-left","--left","x",180,400,1],["drag-right","--right","x",230,470,-1],["drag-bottom","--bottom","y",110,460,-1]].forEach(([id,varname,axis,min,max,direction])=>{
   const node=$(id);let start=null;
   node.addEventListener("pointerdown",e=>{
     const current=parseInt(getComputedStyle(css).getPropertyValue(varname))||parseInt(getComputedStyle(css).getPropertyValue(varname).trim());
     start={coord:axis==="x"?e.clientX:e.clientY,value:current};
     node.setPointerCapture(e.pointerId);node.classList.add("active")
   });
   node.addEventListener("pointermove",e=>{
     if(!start)return;
     const delta=(axis==="x"?e.clientX:e.clientY)-start.coord;
     css.style.setProperty(varname,Math.max(min,Math.min(max,start.value+delta*direction))+"px");
     requestAnimationFrame(drawStructure)
   });
   const end=()=>{start=null;node.classList.remove("active")};
   node.addEventListener("pointerup",end);node.addEventListener("pointercancel",end)
 })
}

// Orthographic interactive crystal projector: no inferred bonds, no fake energy levels.
const cv=$("crystal-canvas"),ctx=cv.getContext("2d");
const colors={Si:"#c59a50",In:"#3f8bad",Sb:"#ba7756",Ga:"#7796cf",As:"#936da3",Ge:"#a486ae",Mg:"#6ab1a5",O:"#c85b56",C:"#627d8e",N:"#5b8fc4",H:"#d3d9e1",Al:"#9cb5ba",S:"#d7b45b",P:"#c17d50"};
function drawStructure(){
 const rect=cv.getBoundingClientRect(),w=Math.max(1,rect.width),h=Math.max(1,rect.height);const dpi=Math.min(window.devicePixelRatio||1,2);
 cv.width=Math.round(w*dpi);cv.height=Math.round(h*dpi);ctx.setTransform(dpi,0,0,dpi,0,0);
 ctx.clearRect(0,0,w,h);
 ctx.strokeStyle="#dbe3eb";ctx.lineWidth=.6;
 for(let x=0;x<w;x+=28){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}
 for(let y=0;y<h;y+=28){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
 const p=S.preview;if(!p)return;
 const c=p.cell;
 const corner=[];for(let bit=0;bit<8;bit++){
   const v=[0,0,0];for(let k=0;k<3;k++)if(bit&(1<<k))for(let a=0;a<3;a++)v[a]+=c[k][a];
   corner.push(v)
 }
 const points=[...corner,...p.positions];const center=[0,1,2].map(i=>(Math.max(...points.map(v=>v[i]))+Math.min(...points.map(v=>v[i])))/2);
 const extent=Math.max(1,...[0,1,2].map(i=>Math.max(...points.map(v=>v[i]))-Math.min(...points.map(v=>v[i]))));
 const scale=Math.min(w,h)*.72/extent*S.zoom;
 function projection(v){
   const [x,y,z]=v.map((s,i)=>s-center[i]);const a=S.angleZ,b=S.angleX;
   const nx=x*Math.cos(a)-y*Math.sin(a),ny=x*Math.sin(a)+y*Math.cos(a);
   const ry=ny*Math.cos(b)-z*Math.sin(b),rz=ny*Math.sin(b)+z*Math.cos(b);
   const perspective=Math.max(.5,Math.min(1.4,1/(1+rz/(extent*5))));
   return{x:w*.5+nx*scale*perspective,y:h*.5-ry*scale*perspective,z:rz,fac:perspective}
 }
 const verts=corner.map(projection);
 if($("viewer-cell").checked){
  ctx.lineWidth=1.35;ctx.strokeStyle="#7194b7b8";
  for(let i=0;i<8;i++)for(let a=0;a<3;a++)if(!(i&(1<<a))){
    const q=verts[i],r=verts[i|(1<<a)];
    ctx.beginPath();ctx.moveTo(q.x,q.y);ctx.lineTo(r.x,r.y);ctx.stroke()
  }
 }
 const atoms=p.positions.map((v,i)=>({...projection(v),sym:p.symbols[i]})).sort((a,b)=>a.z-b.z);
 for(const a of atoms){
  const r=Math.max(2.5,Math.min(12,6.7*a.fac*S.zoom));
  ctx.beginPath();ctx.ellipse(a.x+1.2,a.y+2,r,r*.78,0,0,Math.PI*2);
  ctx.fillStyle="#536a8024";ctx.fill();
  const grad=ctx.createRadialGradient(a.x-r*.32,a.y-r*.45,r*.1,a.x,a.y,r*1.3);
  grad.addColorStop(0,"#ffffff");grad.addColorStop(.28,colors[a.sym]||"#7397a8");grad.addColorStop(1,"#354f6b");
  ctx.beginPath();ctx.arc(a.x,a.y,r,0,Math.PI*2);ctx.fillStyle=grad;ctx.fill();ctx.strokeStyle="#3c587555";ctx.lineWidth=.7;ctx.stroke();
  if($("viewer-labels").checked&&p.positions.length<=140){
    ctx.font="10px Segoe UI";ctx.fillStyle="#37546b";ctx.textAlign="left";ctx.fillText(a.sym,a.x+r+3,a.y+3)
  }
 }
 if($("viewer-axes").checked){
   const origin={x:52,y:h-45};
   const unit=[[1,0,0],[0,1,0],[0,0,1]];
   const vectors=unit.map(v=>{const a=S.angleZ,b=S.angleX,nx=v[0]*Math.cos(a)-v[1]*Math.sin(a),ny=v[0]*Math.sin(a)+v[1]*Math.cos(a);return{dx:nx*28,dy:-(ny*Math.cos(b)-v[2]*Math.sin(b))*28}});
   ["#ba6662","#57a284","#568ec3"].forEach((color,i)=>{const v=vectors[i];ctx.beginPath();ctx.moveTo(origin.x,origin.y);ctx.lineTo(origin.x+v.dx,origin.y+v.dy);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();ctx.fillStyle=color;ctx.font="bold 10px Consolas";ctx.fillText("XYZ"[i],origin.x+v.dx+5,origin.y+v.dy)})
 }
}
cv.addEventListener("pointerdown",e=>{S.dragging={x:e.clientX,y:e.clientY};cv.setPointerCapture(e.pointerId)});
cv.addEventListener("pointermove",e=>{if(!S.dragging)return;S.angleZ+=(e.clientX-S.dragging.x)*.012;S.angleX=Math.max(-1.46,Math.min(1.46,S.angleX+(e.clientY-S.dragging.y)*.012));S.dragging={x:e.clientX,y:e.clientY};drawStructure()});
cv.addEventListener("pointerup",()=>S.dragging=null);cv.addEventListener("pointercancel",()=>S.dragging=null);
cv.addEventListener("wheel",e=>{e.preventDefault();S.zoom=Math.min(4,Math.max(.15,S.zoom*(e.deltaY>0?.9:1.1)));drawStructure()},{passive:false});
const observer=new ResizeObserver(()=>requestAnimationFrame(drawStructure));observer.observe(cv);
function resetView(){S.angleX=-.47;S.angleZ=.64;S.zoom=1;drawStructure()}
async function refreshAll(){
 if(S.loopBusy)return;
 S.loopBusy=true;
 try{await Promise.all([refreshMaterials(),refreshEnvironment()]);await refreshJobs(true);await refreshGpuMonitor();setStatus("工作台数据已更新")}
 catch(e){toast("刷新失败："+e.message,"error")}
 finally{S.loopBusy=false}
}
async function tick(){
 if(S.loopBusy)return;
 S.loopBusy=true;
 try{await refreshJobs(false);await refreshGpuMonitor()}catch(e){console.warn("Polling stopped/failed:",e)}
 finally{S.loopBusy=false}
}
function bind(){
 document.querySelectorAll(".editor-tab").forEach(x=>x.onclick=()=>activateView(x.dataset.view));
 document.querySelectorAll(".dock-tab").forEach(x=>x.onclick=()=>activateDock(x.dataset.dock));
 document.querySelectorAll(".detail-tab").forEach(x=>x.onclick=()=>{S.detail=x.dataset.detail;renderDetail()});
 document.querySelectorAll(".menu-trigger").forEach(x=>x.onclick=()=>toggleMenu(x.dataset.menu));
 document.addEventListener("click",e=>{if(!e.target.closest(".menubar")&&!e.target.closest("#menu-overlay"))hideMenus()});
 document.addEventListener("keydown",handleKeys);
 $("modal-close").onclick=closeModal;$("modal-dismiss").onclick=closeModal;
 $("modal-backdrop").onclick=e=>{if(e.target===$("modal-backdrop"))closeModal()};
 $("tool-import").onclick=chosenFile;$("tool-example").onclick=openSampleDialog;
 $("tool-run").onclick=submitJob;$("inspector-run").onclick=submitJob;
 $("tool-stop").onclick=stopJob;$("tool-retry").onclick=duplicateJob;
 $("tool-env").onclick=showEnvironment;$("tool-report").onclick=report;
 $("report-save").onclick=saveReport;
 $("report-print").onclick=()=>{if(!$("report-print").disabled){activateView("report");requestAnimationFrame(()=>window.print())}};
 $("tool-refresh").onclick=refreshAll;$("explorer-refresh").onclick=refreshMaterials;
 $("material-filter").oninput=renderMaterials;$("compute-category").onchange=updateCalculationCategory;$("backend-select").onchange=setBackendUI;$("abacus-path").oninput=()=>{S.abacusEdited=true;setBackendUI()};
 $("show-workspace").onclick=showEnvironment;
 $("file-input").onchange=e=>uploadFile(e.target.files[0]);
 $("view-reset").onclick=resetView;
 for(const id of ["viewer-cell","viewer-axes","viewer-labels"])$(id).onchange=drawStructure;
 $("result-raw").onclick=rawResult;$("result-export").onclick=exportResultCSV;
 $("compare-export").onclick=exportCompareCSV;
 $("artifacts-refresh").onclick=loadArtifacts;
 $("jobs-filter").onchange=renderJobsTable;$("jobs-search").oninput=renderJobsTable;
 $("dock-clear").onclick=()=>{$("console").textContent="[GUI] 已清空显示；磁盘上的 console.log 未删除。";S.logSize=-1};
 $("materials-folder").onclick=()=>$("materials-list").classList.toggle("closed");
 $("samples-folder").onclick=()=>$("samples-list").classList.toggle("closed");
 // Drag-and-drop structure input on the entire app, without navigating away.
 document.addEventListener("dragover",e=>{if(e.dataTransfer?.types?.includes("Files"))e.preventDefault()});
 document.addEventListener("drop",e=>{if(e.dataTransfer?.files?.length){e.preventDefault();uploadFile(e.dataTransfer.files[0])}});
 makeSplitters();
}
async function start(){
 bind();
 ensureGpuMonitor();
 updateCalculationCategory();
 try{await refreshAll();setStatus("连接成功 · 计算后端就绪")}
 catch(e){toast("连接本地 GUI 服务失败："+e.message,"error");setStatus("连接失败")}
 drawStructure();setInterval(tick,1750);
}
start();
