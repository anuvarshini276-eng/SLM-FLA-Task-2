const $ = id => document.getElementById(id);
let result = null, step = 0, timer = null, revision = 0;
function pause() { clearInterval(timer); timer = null; $('play').textContent = '▶ Play'; }
function draw() {
  if (!result) return;
  const trace = result.exact.trace, state = trace[step].state;
  $('q0').classList.toggle('active', state === 'q0');
  $('q1').classList.toggle('active', state === 'q1');
  $('step-label').textContent = `Step ${step} / ${trace.length - 1}`;
  $('transition').textContent = step === 0 ? 'Start → q0. No symbols read yet.' : `δ(${trace[step - 1].state}, ${trace[step].symbol}) = ${state}${step === trace.length - 1 ? (result.exact.accepted ? ' · ACCEPT' : ' · REJECT') : ''}`;
  $('previous').disabled = step === 0;
  $('next').disabled = step >= trace.length - 1;
  $('reset').disabled = step === 0;
  $('play').disabled = trace.length === 1;
  [...$('tape').children].forEach((node, i) => {node.classList.toggle('read', i < step);node.classList.toggle('current', i === step - 1);});
  const current = $('tape').querySelector('.current');
  if (current) $('tape').scrollLeft = Math.max(0, current.offsetLeft - $('tape').offsetLeft - $('tape').clientWidth / 2);
}
function invalidate() {
  revision++; pause(); result = null;
  $('count').textContent = `${$('string').value.length} / 256`;
  $('neural').textContent = $('exact').textContent = '—';
  $('probability').textContent = 'Run the updated string'; $('final-state').textContent = 'The formal answer';
  $('meter').style.width = '0'; $('agreement').textContent = 'Input changed. Run again to update both answers.';
  $('length-note').textContent = ''; $('error').textContent = '';
  $('trace').replaceChildren(); $('tape').replaceChildren();
  ['reset','previous','next','play'].forEach(id => $(id).disabled = true);
  $('q0').classList.add('active'); $('q1').classList.remove('active');
  $('step-label').textContent = 'Step 0 / 0'; $('transition').textContent = 'Run a string to inspect its state transitions.';
}
async function run(event) {
  event?.preventDefault(); pause();
  const text = $('string').value, requestRevision = ++revision;
  $('error').textContent = '';
  if (!/^[01]*$/.test(text) || text.length > 256) { invalidate(); $('error').textContent = 'Enter only 0 and 1 (up to 256 characters). Empty input is allowed.'; return; }
  $('submit').disabled = true;
  try {
    const response = await fetch('/api/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({string:text})});
    const data = await response.json();
    if (requestRevision !== revision) return;
    if (!response.ok) throw new Error(data.error || 'Prediction failed.');
    result = data; step = 0;
    $('neural').textContent = data.neural ? (data.neural.accepted ? 'Accepted' : 'Rejected') : 'Unavailable';
    $('exact').textContent = data.exact.accepted ? 'Accepted' : 'Rejected';
    $('final-state').textContent = `Final state: ${data.exact.final_state} · exact result`;
    $('probability').textContent = data.neural ? `P(accept) = ${(100*data.neural.accept_probability).toFixed(2)}%` : data.model_message;
    $('meter').style.width = data.neural ? `${100*data.neural.accept_probability}%` : '0';
    $('agreement').textContent = data.neural ? (data.neural.agrees ? '✓ Both answers agree on this string.' : '⚠ Disagreement. Use the exact DFA result.') : 'DFA validation is available; neural prediction needs a model.';
    $('length-note').textContent = data.outside_training_lengths ? 'This length is outside the training range (0–24). Neural generalization is not guaranteed.' : 'Neural probability is a model score, not a guarantee.';
    $('tape').replaceChildren();
    for (const bit of text || 'ε') {const node = document.createElement('span');node.className='bit';node.textContent=bit;$('tape').append(node);}
    $('trace').replaceChildren();
    for (const row of data.exact.trace) {const tr=document.createElement('tr');for (const value of [row.step,row.symbol ?? 'ε (start)',row.state]) {const td=document.createElement('td');td.textContent=value;tr.append(td);}$('trace').append(tr);}
    draw();
  } catch (error) { if (requestRevision === revision) {invalidate();$('error').textContent = `Could not run experiment: ${error.message}`;} }
  finally {$('submit').disabled=false;}
}
$('form').addEventListener('submit',run);
$('string').addEventListener('input',invalidate);
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click',()=>{$('string').value=button.dataset.example;invalidate();run();}));
$('reset').addEventListener('click',()=>{pause();step=0;draw();});
$('previous').addEventListener('click',()=>{pause();if(step>0)step--;draw();});
$('next').addEventListener('click',()=>{pause();if(result&&step<result.string.length)step++;draw();});
$('play').addEventListener('click',()=>{if(timer){pause();return;}if(!result)return;if(step===result.string.length)step=0;draw();$('play').textContent='Ⅱ Pause';timer=setInterval(()=>{step++;draw();if(step>=result.string.length)pause();},650);});
fetch('/api/metrics').then(r=>r.json()).then(m=>{$('metrics').textContent=m.test ? `${(m.test.accuracy*100).toFixed(1)}% on ${m.test.count} unseen strings (lengths 1–24). ${(m.long_test.accuracy*100).toFixed(1)}% on ${m.long_test.count} longer strings (25–128).` : 'Train the model to generate evaluation metrics.';}).catch(()=>{$('metrics').textContent='Evaluation metrics unavailable.';});
run();
