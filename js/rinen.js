/**
 * 設立理念 — 場面を順に送る
 * ------------------------------------------------------------------------
 * お預かりしたものをほぼそのまま使っている。変えたのは、やり直しボタンが
 * 無くても落ちないようにしたところだけ。
 *
 *   ・場面ごとに決めた秒数で自動的に次へ
 *   ・最後の1枚は、理念の全文が下から上へ流れ終わるまで
 *   ・画面を押すと次へ。← → でも送れる。空白キーで止まる
 */
(function(){
  const READ_SEC = 14;   /* 本文が下から現れ、上へ抜け切るまでの秒数 */
  const frame = document.getElementById('frame');
  const slides = [...document.querySelectorAll('.slide')];
  const layers = [...document.querySelectorAll('.sky-layer')];
  const lanterns = [...document.querySelectorAll('.lantern')];
  let idx = -1, timerId = null, tStart = 0, tLeft = 0, paused = false;

  /* 文字サイズの基準：フレーム幅の1% を --u に */
  function setUnit(){
    frame.style.setProperty('--fw', frame.clientWidth + 'px');
    const s = slides[idx];
    if(s && s.classList.contains('sl-read')) measureRoll(s);
  }
  if(window.ResizeObserver) new ResizeObserver(setUnit).observe(frame);
  addEventListener('resize', setUnit);
  setUnit();

  /* 一文字ずつに分解。行をまたいでも順番は続ける */
  slides.forEach(s => {
    let n = 0;
    s.querySelectorAll('[data-split]').forEach(el => {
      const chars = [...el.textContent];
      el.textContent = '';
      chars.forEach(c => {
        const sp = document.createElement('span');
        sp.className = 'ch'; sp.textContent = c; sp.style.setProperty('--i', n++);
        el.appendChild(sp);
      });
    });
  });

  function clearTimer(){ if(timerId){ clearTimeout(timerId); timerId = null; } }
  function startTimer(ms){ clearTimer(); tLeft = ms; tStart = performance.now(); timerId = setTimeout(next, ms); }

  function measureRoll(slide){
    const panel = slide.querySelector('.panel'), roll = slide.querySelector('.roll');
    /* 画面の下の外から現れ、上の外へ抜け切るまでを一続きに */
    const ph = panel.clientHeight, ch = roll.scrollHeight;
    /* まだ組み上がっていないうちに測ると 0 が返る。次の描画で測り直す */
    if(!ph || !ch){ requestAnimationFrame(() => measureRoll(slide)); return; }
    roll.style.setProperty('--from', ph + 'px');
    roll.style.setProperty('--to', (-ch) + 'px');
    roll.style.setProperty('--dur', READ_SEC + 's');
  }

  function go(i){
    if(i === idx) return;
    if(i >= slides.length){ finish(); return; }
    if(i < 0) i = 0;
    clearTimer();
    frame.classList.remove('ending');

    slides.forEach(s => s.classList.remove('active'));
    const s = slides[i];
    void s.offsetWidth;                              /* アニメーションを頭から */
    s.classList.add('active');

    for(let k = 0; k < 6; k++) frame.classList.remove('stage-' + k);
    frame.classList.add('stage-' + i);
    layers.forEach(l => l.classList.toggle('on', +l.dataset.s === i));

    const lv = +(s.dataset.lv || 0);
    lanterns.forEach(l => l.classList.toggle('lit', +l.dataset.lv <= lv));

    idx = i;

    if(s.classList.contains('sl-read')){
      measureRoll(s);
      const roll = s.querySelector('.roll');
      /* animationend は下から上がってくる。中の要素のアニメーションを
         拾って途中で終わらせないよう、本文そのものの終わりだけを見る。
         届かないことがあっても止まらないよう、時間でも受ける。 */
      roll.addEventListener('animationend', function onEnd(e){
        if(e.target !== roll || e.animationName !== 'roll') return;
        roll.removeEventListener('animationend', onEnd);
        if(s.classList.contains('active')) finish();
      });
      startTimer(READ_SEC * 1000 + 900);
    } else {
      startTimer(+s.dataset.dur || 2200);
    }
  }

  function next(){ go(idx + 1); }
  function prev(){ go(Math.max(0, idx - 1)); }
  function finish(){
    clearTimer();
    slides.forEach(s => s.classList.remove('active'));
    frame.classList.add('ending');
    lanterns.forEach(l => l.classList.add('lit'));
    idx = slides.length;
  }

  function setPaused(v){
    paused = v;
    frame.classList.toggle('paused', v);
    if(v){
      if(timerId){ tLeft -= performance.now() - tStart; clearTimer(); }
    } else if(tLeft > 0 && idx < slides.length && !slides[idx]?.classList.contains('sl-read')){
      tStart = performance.now();
      timerId = setTimeout(next, Math.max(0, tLeft));
    }
  }

  const again = document.getElementById('again');
  if(again) again.addEventListener('click', () => { setPaused(false); idx = -1; go(0); });

  document.addEventListener('keydown', e => {
    if(e.key === 'ArrowRight'){ next(); }
    else if(e.key === 'ArrowLeft'){ prev(); }
    else if(e.key === ' '){ e.preventDefault(); setPaused(!paused); }
    else if(e.key === 'g' || e.key === 'G'){ document.body.classList.toggle('guide'); }
  });

  /* 押したときの振る舞い。ふつうは次の場面へ送る。
     ただし本文が流れている最後の1枚だけは、送らずに止める。
     会員は70代が中心で、14秒で読み切れないことがある。送ってしまうと
     終わりの画面へ行ってしまい、本文へ戻る手立てが「もう一度」しかない。 */
  document.getElementById('stage').addEventListener('click', () => {
    const s = slides[idx];
    if(s && s.classList.contains('sl-read')){ setPaused(!paused); return; }
    next();
  });

  /* 書体が届くのを待って始める。ただし待ちきりにはしない。回線が細いときや
     Google の書体に届かないときに、いつまでも何も出ないことになる
     （手元では 12 秒待たされた）。1.2 秒で見切って始め、書体は届いた時点で
     差し替わる。 */
  let booted = false;
  const boot = () => {
    if(booted) return;
    booted = true;
    setUnit();
    setTimeout(() => go(0), 300);
  };
  if(document.fonts && document.fonts.ready) document.fonts.ready.then(boot);
  setTimeout(boot, 1200);
})();
