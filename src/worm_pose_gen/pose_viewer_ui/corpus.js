"use strict";

const corpusUI = (() => {
  let counts={}, samples=[], checkpoints=[], loading=false, source=null;
  function sampleLabel(sample) {
    const basename=(sample.source_path||'').split(/[\\/]/).filter(Boolean).pop()||'Unknown recording';
    const dataset=sample.dataset_path && sample.dataset_path!=='/img_nir' ? ` · ${sample.dataset_path}` : '';
    return `${basename}${dataset} · frame ${sample.frame_index} · ${sample.split} · revision ${sample.revision||1}`;
  }
  function render() {
    const filter=$('#corpus-filter').value.toLowerCase();
    const box=$('#corpus-list');box.innerHTML='';
    for(const sample of samples.filter(s=>sampleLabel(s).toLowerCase().includes(filter))){
      const item=document.createElement('div');item.className='item';
      const text=document.createElement('span');text.textContent=sampleLabel(sample);
      const actions=document.createElement('span');
      const open=document.createElement('button');open.textContent='Open';open.onclick=()=>maskEditor.openCorpus(sample.sample_id);
      const remove=document.createElement('button');remove.textContent='Delete';remove.title='Delete this independent corpus sample';
      remove.onclick=async()=>{if(!maskEditor.beforeMutation())return;if(!window.confirm(`Delete ${sampleLabel(sample)} from the corpus?`))return;try{await api(`/api/corpus/labels/${encodeURIComponent(sample.sample_id)}`,{method:'DELETE'});await refresh();}catch(e){setStatus(e.message,'error');}};
      actions.append(open,remove);item.append(text,actions);box.append(item);
    }
    if(!box.children.length)box.textContent=samples.length?'No matching labels.':'No labels yet. Save a workspace override, then Save label to corpus.';
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
    if(loading)return;loading=true;
    try{const payload=await api('/api/corpus');samples=payload.samples||[];counts=payload.counts||{};$('#corpus-train-status').textContent=!counts.train||!counts.val?'Training requires at least one train and one val sample.':$('#corpus-train-status').textContent;$('#corpus-counts').textContent=`${samples.length} labels · ${Object.entries(payload.counts||{}).map(([k,v])=>`${k} ${v}`).join(' · ')} · ${payload.root||''}`;render();await refreshCheckpoints();}
    catch(e){$('#corpus-counts').textContent=serverIsApp()?e.message:'Corpus and training need the app server.';$('#corpus-train').disabled=true;}
    finally{loading=false;}
  }
  async function train() {
    const params={device:$('#tune-device').value,crop_size:Number($('#tune-crop').value),patience:Number($('#tune-patience').value),num_workers:Number($('#tune-workers').value),seed:Number($('#tune-seed').value),epochs:Number($('#tune-epochs').value),batch_size:Number($('#tune-batch').value),learning_rate:Number($('#tune-lr').value)};
    if(!Number.isInteger(params.epochs)||params.epochs<1||!Number.isInteger(params.batch_size)||params.batch_size<1||!Number.isFinite(params.learning_rate)||params.learning_rate<=0){setStatus('Enter positive epochs, batch size and learning rate.','error');return;}
    $('#corpus-train').disabled=true;
    try{const job=await post('/api/jobs',{kind:'fine_tune',params});$('#corpus-train-status').textContent=`Fine-tune ${job.id} queued. Progress, log and cancellation are in Pipeline → Jobs.`;$('#jobs-mine').checked=false;await loadJobs();showTab('pipeline');}
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
    render();
  }
  function init(){
    $('#corpus-refresh').onclick=refresh;$('#corpus-filter').oninput=render;
    $('#checkpoint-refresh').onclick=refreshCheckpoints;$('#checkpoint-select').onclick=selectCheckpoint;$('#corpus-train').onclick=train;
    $('#corpus-close').onclick=()=>{if(maskEditor.leave()){showRow(state.row,{keepView:true,immediate:true});showTab('view');}};
    render();
  }
  return {sampleLabel,init,refresh,refreshCheckpoints,onJobs,sourceChanged};
})();
