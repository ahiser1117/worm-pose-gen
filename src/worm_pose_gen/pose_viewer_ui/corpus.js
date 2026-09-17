"use strict";

const corpusUI = (() => {
  let counts={}, samples=[], checkpoints=[], request=0, source=null, filterTimer=null;
  function sampleLabel(sample) {
    const basename=(sample.source_path||'').split(/[\\/]/).filter(Boolean).pop()||'Unknown recording';
    const dataset=sample.dataset_path && sample.dataset_path!=='/img_nir' ? ` · ${sample.dataset_path}` : '';
    return `${basename}${dataset} · frame ${sample.frame_index} · ${sample.split} · revision ${sample.revision||1}`;
  }
  function getFilters() {
    const ids={source:'corpus-source-filter',split:'corpus-split-filter',recording:'corpus-recording-filter',q:'corpus-filter'};
    return Object.fromEntries(Object.entries(ids).map(([key,id])=>[key,document.getElementById(id)?.value.trim()||'']).filter(([,value])=>value && value!=='all'));
  }
  function fillOptions(id, values, label) {
    const select=document.getElementById(id);if(!select)return;
    const previous=select.value==='all'?'':select.value;select.replaceChildren(new Option(label,''));
    for(const value of values){const v=typeof value==='object'?(value.value||value.id||value.path||value.recording):value;if(!v)continue;select.add(new Option(typeof value==='object'?(value.label||value.name||(value.path?`${value.path} · ${value.dataset||'/img_nir'}`:v)):v,v));}
    // Preserve the selected filter even if no remaining sample has that value.
    if(previous && ![...select.options].some(o=>o.value===previous))select.add(new Option(previous,previous));
    select.value=previous;
  }
  function renderFilters(facets={}) {
    fillOptions('corpus-source-filter',facets.sources||[...new Set(samples.map(s=>s.label_source).filter(Boolean))],'All sources');
    fillOptions('corpus-split-filter',facets.splits||['train','val','test'],'All splits');
    const recording=state.run?.entry?.recording;
    const recordings=(facets.recordings||[...new Set(samples.map(s=>s.source_path).filter(Boolean))]).map(item=>typeof item==='object'&&item.path===recording?{...item,label:`Current recording · ${item.path} · ${item.dataset||'/img_nir'}`}:item);
    fillOptions('corpus-recording-filter',recordings,'All recordings');
  }
  function renderRecordings() {
    const select=document.getElementById('corpus-new-recording');if(!select)return;
    const previous=select.value;select.replaceChildren(new Option('Choose a recording',''));
    for(const rec of state.recordings||[]){const option=new Option(`${rec.path} · ${rec.dataset||'/img_nir'}${rec.readable===false?' · unavailable':''}`,JSON.stringify([rec.path,rec.dataset||'/img_nir']));option.disabled=rec.readable===false;select.add(option);}
    if([...select.options].some(o=>o.value===previous))select.value=previous;
    const button=document.getElementById('corpus-new-open');if(button)button.disabled=!serverIsApp()||!select.value;
  }
  function getPool() {
    const value=document.getElementById('corpus-new-recording')?.value;
    if(!value)return null;
    const [recording,dataset]=JSON.parse(value);
    return {recordings:[{recording,dataset}]};
  }
  async function openNewLabel() {
    const pool=getPool(),frame=Number(document.getElementById('corpus-new-frame')?.value);
    if(!pool||!Number.isInteger(frame)||frame<0){setStatus('Choose a recording and a nonnegative frame number.','error');return;}
    const target={...pool.recordings[0],frame};
    const rec=(state.recordings||[]).find(r=>r.path===target.recording&&(r.dataset||'/img_nir')===target.dataset);
    if(rec && Number.isInteger(rec.frames) && frame>=rec.frames){setStatus(`Frame must be less than ${rec.frames}.`,'error');return;}
    try{const opened=await maskEditor.openTarget(target);if(opened===false)return;maskEditor.setNavigationPool(pool);const note=document.getElementById('corpus-new-status');if(note)note.textContent=`Labeling ${target.recording} · ${target.dataset}. Saves affect the corpus only.`;}
    catch(error){setStatus(error.message,'error');}
  }
  async function browseFiltered() {
    await refresh();
    if(!samples.length){setStatus('No saved labels match these filters.','error');return false;}
    return maskEditor.openCorpus(samples[0].sample_id);
  }
  function render() {
    const box=$('#corpus-list');box.innerHTML='';
    for(const sample of samples){
      const item=document.createElement('div');item.className='item';item.dataset.sample=sample.sample_id;
      const text=document.createElement('span');text.textContent=sampleLabel(sample);
      const metadata=document.createElement('div');metadata.className='meta';metadata.textContent=`${sample.source_path||'Source unavailable'} · ${sample.dataset_path||'/img_nir'} · ${sample.label_source||'saved label'}`;text.append(metadata);
      const actions=document.createElement('span');
      const open=document.createElement('button');open.type='button';open.textContent='Open';open.onclick=()=>maskEditor.openCorpus(sample.sample_id);
      const remove=document.createElement('button');remove.type='button';remove.textContent='Delete corpus label';remove.title='Delete this independent corpus sample';
      remove.onclick=async()=>{if(!await maskEditor.requestLeave())return;if(!window.confirm(`Delete ${sampleLabel(sample)} from the corpus?`))return;try{await api(`/api/corpus/labels/${encodeURIComponent(sample.sample_id)}`,{method:'DELETE'});await refresh();}catch(e){setStatus(e.message,'error');}};
      actions.append(open,remove);item.append(text,actions);box.append(item);
    }
    if(!box.children.length){
      const filtered=Object.keys(getFilters()).length>0;
      const note=document.createElement('p');note.textContent=filtered?'No matching labels. Clear filters to browse saved labels.':'No labels yet. Choose a recording to label a new frame.';
      const action=document.createElement('button');action.type='button';action.textContent=filtered?'Clear filters':'Label a new frame';
      action.onclick=()=>{if(filtered){for(const id of ['corpus-filter','corpus-source-filter','corpus-split-filter','corpus-recording-filter'])$('#'+id).value='';refresh();}else $('#corpus-new-recording').focus();};
      box.append(note,action);
    }
    $('#corpus-browse').disabled=!samples.length;
    const base=checkpoints.find(c=>c.base && c.exists!==false);
    $('#training-counts').textContent=`${counts.train||0} training · ${counts.val||0} validation · ${counts.test||0} test labels`;
    $('#training-readiness').textContent=!serverIsApp()?'Training needs the app server.':!counts.train||!counts.val?'Add at least one training label and one validation label before training.':!base?'The packaged starting model is unavailable. Configure it on the app server, then refresh checkpoints.':'Training data and starting model are available.';
    $('#training-base-model').textContent=base?`Starting model: ${base.label||base.path}`:'Starting model: packaged checkpoint unavailable.';
    $('#corpus-train').disabled=!serverIsApp()||!counts.train||!counts.val||!checkpoints.some(c=>c.base && c.exists!==false);
    $('#checkpoint-select').disabled=!serverIsApp()||!isWorkspace()||!checkpoints.length;
  }
  async function refreshCheckpoints() {
    try{
      const result=await api('/api/checkpoints');checkpoints=(result.checkpoints||[]).filter(c=>c.exists!==false);
      const select=$('#seg-checkpoint'), selected=state.run&&state.run.selected_checkpoint, match=checkpoints.find(c=>c.path===selected), previous=match?(match.id||match.path):select.value;select.innerHTML='';
      for(const checkpoint of checkpoints){const option=document.createElement('option');option.value=checkpoint.id||checkpoint.path;option.textContent=checkpoint.label||checkpoint.path;select.append(option);}
      if([...select.options].some(o=>o.value===previous))select.value=previous;
      $('#checkpoint-status').textContent=checkpoints.length?'Selecting a checkpoint changes future segmentation. Saved overrides remain available.':'No segmentation checkpoint is available. Configure the packaged checkpoint on the app server.';
      render();
    }catch(e){$('#checkpoint-status').textContent=e.message;}
  }
  async function refresh() {
    const token=++request,query=new URLSearchParams(getFilters()).toString();
    try{
      const payload=await api(`/api/corpus${query?'?'+query:''}`);if(token!==request)return;
      samples=payload.samples||[];counts=payload.counts||{};renderFilters(payload.facets||{});

      const total=Object.values(counts).filter(v=>typeof v==='number').reduce((a,v)=>a+v,0);
      $('#corpus-counts').textContent=`${samples.length} matching / ${total} total labels · ${Object.entries(counts).map(([k,v])=>`${k} ${v}`).join(' · ')} · ${payload.root||''}`;
      render();await refreshCheckpoints();
      if(!(state.recordings||[]).length && serverIsApp())await loadRecordings(false);
      renderRecordings();
    }catch(e){if(token!==request)return;$('#corpus-counts').textContent=serverIsApp()?e.message:'Corpus and training need the app server.';$('#corpus-train').disabled=true;}
  }
  async function train() {
    const params={device:$('#tune-device').value,crop_size:Number($('#tune-crop').value),patience:Number($('#tune-patience').value),num_workers:Number($('#tune-workers').value),seed:Number($('#tune-seed').value),epochs:Number($('#tune-epochs').value),batch_size:Number($('#tune-batch').value),learning_rate:Number($('#tune-lr').value)};
    if(!Number.isInteger(params.epochs)||params.epochs<1||!Number.isInteger(params.batch_size)||params.batch_size<1||!Number.isFinite(params.learning_rate)||params.learning_rate<=0){setStatus('Enter positive epochs, batch size and learning rate.','error');return;}
    if(!Number.isInteger(params.crop_size)||params.crop_size<1||!Number.isInteger(params.patience)||params.patience<0||!Number.isInteger(params.num_workers)||params.num_workers<0||!Number.isInteger(params.seed)){setStatus('Crop must be positive; patience and workers nonnegative; seed must be an integer.','error');return;}
    $('#corpus-train').disabled=true;
    try{const job=await post('/api/jobs',{kind:'fine_tune',params});$('#corpus-train-status').textContent=`Fine-tune ${job.id} queued. Progress, log and cancellation are in Jobs.`;$('#jobs-mine').checked=false;await loadJobs();if(typeof openRightTab==='function')openRightTab('jobs');else showTab('training');}
    catch(e){setStatus(e.message,'error');}finally{render();}
  }
  async function selectCheckpoint() {
    if(!maskEditor.beforeMutation())return;
    const chosen=checkpoints.find(c=>(c.id||c.path)===$('#seg-checkpoint').value);if(!chosen)return;
    try{await post(editsUrl('checkpoint'),{checkpoint:chosen.id||chosen.path});
      for(const stage of state.stages||[])if((stage.params||[]).some(p=>p.name==='checkpoint'))state.stageValues[stage.name]={...(state.stageValues[stage.name]||{}),checkpoint:chosen.path};
      state.frameCache.clear();maskEditor.invalidate();await reloadSource();rebuildStages();$('#checkpoint-status').textContent=`Selected ${chosen.label||chosen.path} for ${state.runName}.`;}
    catch(e){setStatus(e.message,'error');}
  }
  function onJobs(jobs) {
    const done=jobs.filter(j=>(j.spec||{}).kind==='fine_tune');
    if(done.length){$('#corpus-train-status').textContent=done.map(j=>`${j.id}: ${j.state}${j.error?' · '+j.error:''}`).join(' · ');refreshCheckpoints();}
  }
  function sourceChanged() {
    const current=currentSourceKey();
    if(current!==source){
      source=current;
      const selected=state.run&&state.run.selected_checkpoint;
      for(const stage of state.stages||[])if((stage.params||[]).some(p=>p.name==='checkpoint')){
        state.stageValues[stage.name]={...(state.stageValues[stage.name]||{})};
        if(selected)state.stageValues[stage.name].checkpoint=selected;else delete state.stageValues[stage.name].checkpoint;
      }
      rebuildStages();
    }
    render();renderRecordings();
  }
  function init(){
    $('#corpus-refresh').onclick=refresh;$('#corpus-filter').oninput=()=>{clearTimeout(filterTimer);filterTimer=setTimeout(refresh,180);};
    for(const id of ['corpus-source-filter','corpus-split-filter','corpus-recording-filter']){const node=document.getElementById(id);if(node)node.onchange=refresh;}
    const recording=document.getElementById('corpus-new-recording');if(recording)recording.onchange=()=>{const button=document.getElementById('corpus-new-open');if(button)button.disabled=!recording.value;};
    const newLabel=document.getElementById('corpus-new-open');if(newLabel)newLabel.onclick=openNewLabel;
    const browse=document.getElementById('corpus-browse');if(browse)browse.onclick=browseFiltered;
    $('#checkpoint-refresh').onclick=refreshCheckpoints;$('#checkpoint-select').onclick=selectCheckpoint;$('#corpus-train').onclick=train;
    $('#corpus-close').onclick=()=>maskEditor.returnToWorkspace();
    $('#training-add-labels').onclick=()=>{showTab('labels');$('#corpus-new-title').focus();};
    $('#training-jobs').onclick=()=>{$('#jobs-mine').checked=false;openRightTab('jobs');};
    render();
  }
  return {sampleLabel,init,refresh,refreshCheckpoints,onJobs,sourceChanged,getFilters,getPool,openNewLabel,renderRecordings,browseFiltered};
})();
