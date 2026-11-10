import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const here=path.dirname(fileURLToPath(import.meta.url));
const repo=path.resolve(here,'..');
const data=JSON.parse(await fs.readFile(path.join(here,'storyboard_data.json'),'utf8'));
if(data.slides.length!==60) throw new Error(`Потрібно рівно 60 слайдів; знайдено ${data.slides.length}`);
const build=path.join(repo,'.presentation-build');
const output=repo;
const renders=path.join(build,'renders');
await fs.mkdir(renders,{recursive:true});
const oldMedia=path.join(here,'assets');
const imgA=new Uint8Array(await fs.readFile(path.join(oldMedia,'frame-A.jpg')));
const imgB=new Uint8Array(await fs.readFile(path.join(oldMedia,'frame-B.jpg')));
const imgD=new Uint8Array(await fs.readFile(path.join(oldMedia,'frame-D.jpg')));
const bg='#091426', panel='#12243D', ink='#EFF5FF', muted='#B5C2D8', lineColor='#314964';
const cols={N:'#A9B5C7',A:'#80BFFF',B:'#F8C76B',C:'#C4B9E7',D:'#55DACB',head:'#E69F68',plan:'#B69AFF'};
const P=Presentation.create({slideSize:{width:1280,height:720}});
const rect=(s,x,y,w,h,fill,r=0,stroke='none')=>s.shapes.add({geometry:r?'roundRect':'rect',position:{left:x,top:y,width:w,height:h},fill,line:{style:'solid',fill:stroke,width:stroke==='none'?0:1},...(r?{borderRadius:r}:{})});
const txt=(s,t,x,y,w,h,size=22,color=ink,bold=false,align='left')=>{const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{style:'solid',fill:'none',width:0}});q.text=t;q.text.style={typeface:'Arial',fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',wrap:true};return q;};
const ln=(s,x1,y1,x2,y2,c=lineColor,w=2)=>s.shapes.add({geometry:'line',position:{left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.abs(x2-x1),height:Math.abs(y2-y1)},line:{style:'solid',fill:c,width:w}});
const dot=(s,x,y,r,c)=>s.shapes.add({geometry:'ellipse',position:{left:x-r,top:y-r,width:r*2,height:r*2},fill:c,line:{style:'solid',fill:'none',width:0}});
const image=(s,blob,x,y,w,h,alt)=>s.images.add({blob,contentType:'image/jpeg',alt,fit:'cover',position:{left:x,top:y,width:w,height:h},geometry:'roundRect',borderRadius:8});
function barChart(s,labels,values,{x=90,y=240,w=1090,h=350,colors=labels.map(x=>cols[x]||cols.head),min=0,max=null,unit='%',baseline=0}={}){
 const plotX=x+155, plotW=w-220, row=h/labels.length; const lim=max||Math.max(...values.map(Math.abs))*1.12;
 if(min<0||values.some(v=>v<0)){
  const zeroY=y+h/2; ln(s,plotX,zeroY,plotX+plotW,zeroY,muted,1);
  labels.forEach((lab,i)=>{const yy=y+i*row+row/2;txt(s,lab,x,yy-16,135,30,17,muted,true);const center=plotX+plotW/2, bw=Math.abs(values[i])/lim*plotW/2;rect(s,values[i]<0?center-bw:center,yy-12,bw,24,colors[i],5);txt(s,`${values[i]>0?'+':''}${String(values[i].toFixed(2)).replace('.',',')} ${unit}`,plotX+plotW+8,yy-16,130,30,16,colors[i],true);});
 }else{
  const base=plotX; ln(s,base,y-8,base,y+h,muted,1);
  labels.forEach((lab,i)=>{const yy=y+i*row+row/2;txt(s,lab,x,yy-16,135,30,17,muted,true);const bw=Math.max(2,values[i]/lim*plotW);rect(s,base,yy-12,bw,24,colors[i],5);txt(s,`${String(values[i].toFixed(2)).replace('.',',')}${unit}`,Math.min(base+bw+10,plotX+plotW-5),yy-16,112,30,16,colors[i],true);});
 }
}
function addImageOrChart(s,n,kind){
 const cx=650, cy=200, cw=550;
 if(kind==='hero'){image(s,imgD,650,170,540,380,'Кадр із реального відеоприкладу моделі D');txt(s,'Детекція птаха · відеоприклад',650,565,540,28,15,muted,false,'right');return;}
 if(['cubBars','videoBars'].includes(kind)){const labs=['Pretrained N','A','B','C','D'];const vals=kind==='cubBars'?[78.3969,81.0817,80.8108,81.0672,81.1704]:[56.3649,50.9690,51.9417,50.6617,51.9841];barChart(s,labs,vals,{colors:[cols.N,cols.A,cols.B,cols.C,cols.D],x:90,y:220,w:1090,h:350,max:kind==='cubBars'?90:65,unit:'%'});txt(s,kind==='cubBars'?'CUB validation · 600 фото · mAP50–95':'Legacy video validation · 32 кадри · mAP50–95',95,590,1000,26,15,muted);return;}
 if(kind==='deltaCub'){barChart(s,['A','B','C','D'],[2.6848,2.4139,2.6703,2.7735],{colors:[cols.A,cols.B,cols.C,cols.D],x:100,y:230,w:1080,h:330,max:3.2,unit:'в.п.'});txt(s,'Δ = адаптована модель − pretrained N',100,590,900,28,16,muted);return;}
 if(kind==='deltaVideo'){barChart(s,['A','B','C','D'],[-5.3959,-4.4232,-5.7032,-4.3808],{colors:[cols.A,cols.B,cols.C,cols.D],x:100,y:230,w:1080,h:330,max:6.2,unit:'в.п.'});txt(s,'Від’ємні значення показано ліворуч від нуля',100,590,900,28,16,muted);return;}
 if(kind==='scatterLegacy'||kind==='scatterV2'){
  const points=kind==='scatterLegacy'?[['N',78.3969,56.3649,cols.N],['A',81.0817,50.9690,cols.A],['B',80.8108,51.9417,cols.B],['C',81.0672,50.6617,cols.C],['D',81.1704,51.9841,cols.D]]:[['N',78.3969,77.2377,cols.N],['D',81.1704,77.9710,cols.D],['Head3',81.0810,78.4981,cols.head],['Head5',81.0932,78.4039,cols.plan]];
  const x0=230,y0=555,ww=830,hh=315;ln(s,x0,y0,x0+ww,y0,muted,1);ln(s,x0,y0,x0,y0-hh,muted,1);
  let [xmin,xmax,ymin,ymax]=kind==='scatterLegacy'?[77.5,82,48,58]:[77,82,70,80];
  points.forEach(([l,x,y,c])=>{const px=x0+(x-xmin)/(xmax-xmin)*ww,py=y0-(y-ymin)/(ymax-ymin)*hh;dot(s,px,py,9,c);const dy=l==='Head3'?-40:l==='Head5'?14:l==='A'?-42:l==='C'?10:-17;txt(s,l,px+12,py+dy,90,26,16,c,true);});
  txt(s,'CUB mAP50–95 (%)',500,585,300,26,16,muted,false,'center');txt(s,kind==='scatterLegacy'?'Legacy video mAP50–95 (%)':'V2 video mAP50–95 (%)',78,196,140,46,14,muted);txt(s,kind==='scatterLegacy'?'CUB validation 600 · legacy video 32':'V2: development-validation · фото: CUB validation',285,608,840,20,13,muted);return;
 }
 if(kind==='daCompare'){barChart(s,['CUB · D−A','Legacy video · D−A'],[.0886,1.0151],{colors:[cols.D,cols.D],x:100,y:245,w:1060,h:250,max:1.15,unit:'в.п.'});txt(s,'Малий zoom лише для історичного D проти A',100,570,900,26,15,muted);return;}
 if(kind==='studyAll'){barChart(s,['Raw N → head','D → head3','D → neck+head','Raw S → head','D → head5'],[75.9916,78.4981,78.0153,71.4432,78.4039],{colors:[cols.N,cols.head,cols.plan,cols.C,cols.D],x:70,y:205,w:1120,h:340,max:82,unit:'%'});txt(s,'Стандартний evaluator · v2 development-validation · власні старти див. на наступному слайді',75,580,1070,26,14,muted);return;}
 if(kind==='studyDelta'){barChart(s,['Raw N → head','D → head3','D → neck+head','Raw S → head','D → head5'],[-1.2461,.5271,.0443,.2457,.4329],{colors:[cols.N,cols.head,cols.plan,cols.C,cols.D],x:70,y:205,w:1120,h:340,max:1.4,unit:'в.п.'});txt(s,'Δ = trained checkpoint − власний epoch 0',75,580,1070,26,14,muted);return;}
 if(kind==='studyStart'){barChart(s,['Raw N · epoch 0','Raw N · trained','D · epoch 0','D · head3'],[77.2377,75.9916,77.9710,78.4981],{colors:[cols.N,cols.N,cols.D,cols.head],x:80,y:220,w:1080,h:300,max:82,unit:'%'});txt(s,'N: −1,2461 в.п. до власного старту',125,565,480,32,17,cols.N,true);txt(s,'D: +0,5271 в.п. до власного старту',650,565,500,32,17,cols.head,true);return;}
 if(kind==='photoVideo'){barChart(s,['D','Head3','Head5'],[77.9710,78.4981,78.4039],{colors:[cols.D,cols.head,cols.plan],x:80,y:220,w:500,h:310,max:82,unit:'%'});barChart(s,['D','Head3','Head5'],[81.1704,81.0810,81.0932],{colors:[cols.D,cols.head,cols.plan],x:650,y:220,w:500,h:310,max:83,unit:'%'});txt(s,'V2 video · mAP50–95',100,570,470,28,16,muted,false,'center');txt(s,'CUB photo · mAP50–95',670,570,470,28,16,muted,false,'center');return;}
 if(kind==='tilesAP'){barChart(s,['F0 · full frame','F1 · full + tiles'],[74.9868,41.5870],{colors:[cols.D,cols.plan],x:100,y:250,w:1080,h:260,max:80,unit:'%'});return;}
 if(kind==='tilesFPS'){barChart(s,['F0 · 32 calls','F1 · 176 calls'],[10.79,1.37],{colors:[cols.D,cols.plan],x:100,y:230,w:1080,h:250,max:12,unit:' кадрів/с'});txt(s,'Один paired protocol F · 32 зображення',100,560,950,28,16,muted);return;}
 if(kind==='seedResults'){barChart(s,['CUB · середнє D','Legacy video · середнє D'],[81.22,51.51],{colors:[cols.A,cols.D],x:100,y:220,w:1080,h:220,max:90,unit:'%'});txt(s,'Вибіркове SD: 0,04 в.п. CUB · 0,38 в.п. video · 5 запусків',100,525,1050,34,18,muted);return;}
 if(kind==='benchmark'){barChart(s,['CPU · перша епоха','MPS · перша епоха','CPU · прогріта','MPS · прогріта'],[31.48,34.97,24.98,27.36],{colors:[cols.A,cols.D,cols.A,cols.D],x:80,y:220,w:1080,h:290,max:40,unit:' с'});txt(s,'Час епохи · один контроль',80,560,1000,28,15,muted);return;}
 if(kind==='throughput'){barChart(s,['CPU · прогріта','MPS · прогріта'],[5.12,4.68],{colors:[cols.A,cols.D],x:90,y:230,w:1080,h:270,max:6,unit:' зобр./с'});txt(s,'128 train · 64 validation · 3 епохи · один парний замір',100,560,1050,28,15,muted);return;}
 if(kind==='memory'){barChart(s,['RSS · CPU','RSS · MPS','MPS allocated','MPS driver'],[3024,3490,144.5,1199.5],{colors:[cols.A,cols.D,cols.B,cols.plan],x:80,y:220,w:1080,h:290,max:3900,unit:' MiB'});txt(s,'Різні лічильники; MPS allocated/driver не додаються до RSS',80,560,1050,30,15,muted);return;}
 if(kind==='gt'){
  txt(s,'Legacy GT',680,240,220,32,20,'#F8C76B',true,'center');
  txt(s,'V2 GT',950,240,220,32,20,cols.D,true,'center');
  rect(s,700,315,180,150,'none',0,'#F8C76B');rect(s,970,295,180,190,'none',0,cols.D);
  txt(s,'рамка однієї цілі',680,490,220,28,16,muted,false,'center');txt(s,'усі видимі частини',950,510,220,28,16,muted,false,'center');
  txt(s,'Схема визначення рамки · не кадр із прогнозом',665,562,500,24,14,muted,false,'right');return;
 }
 if(kind==='videoDemo'){image(s,imgA,665,210,500,280,'Реальний кадр прикладу детекції');txt(s,'Нерозмічена демонстрація · не метрика якості',665,520,500,26,14,muted,false,'right');return;}
 if(kind==='background'){txt(s,'0 / 8',cx,250,cw,110,76,cols.D,true,'center');txt(s,'фонів із хибними детекціями',cx,370,cw,40,22,ink,true,'center');txt(s,'D · Head3 · Head5 · confidence ≥ 0,25',cx,430,cw,32,17,muted,false,'center');txt(s,'Невеликий контроль; не повні негативні ролики',cx,475,cw,32,15,muted,false,'center');return;}
 if(kind==='seeds'){
  const rows=[['CUB mAP50–95',81.17036776208397,81.26527055947654,[81.17036776208397,81.18772249584897,81.26527055947654,81.26074062762264,81.22907447948965],cols.A],['Legacy video mAP50–95',51.03520402019124,51.98413335002593,[51.98413335002593,51.76127597189681,51.24099307961666,51.55165711026317,51.03520402019124],cols.D]];
  rows.forEach(([lab,min,max,vals,c],ri)=>{const yy=300+ri*130,x0=760,w=390;txt(s,lab,670,yy-36,470,27,17,muted,true);ln(s,x0,yy,x0+w,yy,muted,1);vals.forEach((v,i)=>{const xx=x0+(v-min)/(max-min)*w;dot(s,xx,yy,7,c);txt(s,String(26+i),xx-13,yy+12,30,22,13,c,true,'center');});});
  txt(s,'Кожна точка — окремий seed 26–30',670,548,480,22,14,muted,false,'center');
  txt(s,'Середнє / вибіркове SD: CUB 81,22% / 0,04 в.п. · video 51,51% / 0,38 в.п.',650,576,530,40,14,ink,true,'center');return;
 }
 if(kind==='counts'){
  const nums=n===7?[5377,600,499]:n===8?[87,24,8]:n===23?[24,8]:n===25?[102,15,32,164]:[3,2,0];
  const labels=n===7?['Train','Validation','Test']:n===8?['Train · загалом','Validation · позитивні','Фонові вирізки']:n===23?['Позитивні v2 кадри','Фонові вирізки']:n===25?['Унікальні відеоприклади','Кадрів ×2 додаткові повтори','CUB фото','Пред’явлення / епоху']:['Train sequences','Validation sequences','Збігів у перевірці'];
  const y=240,row=70;labels.forEach((lab,i)=>{const yy=y+i*row;txt(s,lab,670,yy,355,36,18,muted,true);rect(s,670,yy+42,Math.min(430,nums[i]/Math.max(...nums)*430),12,[cols.A,cols.D,cols.plan,cols.head][i%4],5);txt(s,String(nums[i]),1110,yy+24,90,34,24,[cols.A,cols.D,cols.plan,cols.head][i%4],true,'right');});return;
 }
 if(kind==='flow'||kind==='lineage'||kind==='variants'||kind==='protocol'||kind==='layers'||kind==='tiles'||kind==='decision'||kind==='roadmap'||kind==='plan'||kind==='audit'||kind==='sources'||kind==='baseline'||kind==='metric'||kind==='errors'||kind==='matrix'||kind==='limits'||kind==='summary'||kind==='labelAudit'||kind==='v2Annotation'||kind==='specs'||kind==='formula'||kind==='commands'){
  if(kind==='lineage'){
   // A two-dimensional parent/child map. Connectors stop at opaque cards so
   // no route runs across candidate markers or labels.
   const node=(x,y,w,h,title,detail,color)=>{
    rect(s,x,y,w,h,panel,10,lineColor);
    rect(s,x,y,6,h,color,3);
    txt(s,title,x+18,y+7,w-28,25,17,color,true);
    txt(s,detail,x+18,y+32,w-28,20,13,muted);
   };
   ln(s,850,372,880,372,lineColor,2);
   ln(s,1030,372,1046,372,lineColor,2);
   ln(s,1046,275,1046,465,lineColor,2);
   ln(s,1046,275,1065,275,lineColor,2);
   ln(s,1046,370,1065,370,lineColor,2);
   ln(s,1046,465,1065,465,lineColor,2);
   node(680,340,170,64,'Pretrained N','вихідні ваги',cols.N);
   node(880,340,150,64,'A','фотоорієнтир',cols.A);
   node(1065,245,165,60,'B','від A',cols.B);
   node(1065,340,165,60,'C','від A',cols.C);
   node(1065,435,165,60,'D','від A',cols.D);
   return;
  }
  let labels=[];
  if(kind==='flow')labels=['Вхідний MP4','Покадровий inference','BBox · class · conf','Вихідний MP4'];
  if(kind==='lineage')labels=['Pretrained N','A · фотоорієнтир',['B · від A','C · від A','D · від A']];
  if(kind==='variants')labels=['A · 87 відео + 128 фото','B · від A','C · нижчий LR','D · 512 фото · більше кроків'];
  if(kind==='protocol')labels=['Стартові ваги','Freeze / unfreeze','3 або 5 епох','v2 validation'];
  if(kind==='layers')labels=['Backbone · frozen','Neck · frozen / open','Head · trainable','BN · frozen'];
  if(kind==='tiles')labels=['Full frame','Tiles 640 · overlap 20%','Перенесення координат','Global NMS'];
  if(kind==='decision')labels=['Приріст є у своїй вибірці','Перевірити інші метрики','Мала вибірка й один seed','Залишити як експеримент'];
  if(kind==='roadmap'||kind==='plan')labels=['Нові незалежні сцени','Повна GT','Повтори head + photo guard','Go / no-go за метриками'];
  if(kind==='audit')labels=['Model selection','COCO → bird remap','Protected files','46 tests · audit commit'];
  if(kind==='sources')labels=['Первинні JSON / CSV','Протоколи й ваги','README узгодження','SHA256 provenance'];
  if(kind==='baseline')labels=['COCO pretrained','До fine-tuning','A вже адаптована','Δ рахуємо від N'];
  if(kind==='metric')labels=['mAP50–95','Абсолютно: %','Різниця: в.п.','Validation ≠ test'];
  if(kind==='errors')labels=['FP / FN лише з GT','Demo ≠ AP / recall','Proxy labels не GT','Позначити межі висновку'];
  if(kind==='matrix')labels=['Head · виміряно','Neck · без переваги','S · нижчий власний старт','Tiles · відхилено'];
  if(kind==='limits')labels=['24 positive + 8 background','2 known sequences','1 seed у новій серії','Незалежний test: не виміряно'];
  if(kind==='summary')labels=['Pretrained як baseline','A–D · legacy спад відео','Head3 · малий v2 плюс','D лишається release'];
  if(kind==='labelAudit')labels=['Формат рамок перевірено','Дублікати та повтори перевірено','Згруповано за послідовностями','Ризик витоку знижено, не усунено'];
  if(kind==='v2Annotation')labels=['Розмічали без прогнозів моделі','Охоплено видимі частини птаха','Незалежного людського аудиту немає','Development-validation, не test'];
  if(kind==='specs')labels=['CPU · i7-8750H','Radeon Pro 555X · 4 GB','32 GB RAM','macOS 15.8.1'];
  if(kind==='formula')labels=['102 унікальних video','15 × (3−1) повтори','32 фото CUB','164 пред’явлення'];
  if(kind==='commands')labels=['D · weights/best.pt','Head3 · weights/video_focus.pt','Сценарій у README','Різні імена запусків'];
  if(kind==='protocol'){
   const routes=[['Pretrained N → head · 3 ep','−1,2461 в.п.',cols.N],['D → head · 3 ep','+0,5271 в.п.',cols.head],['D → neck + head · 3 ep','+0,0443 в.п.',cols.plan],['Pretrained S → head · 3 ep','+0,2457 в.п.',cols.C],['D → head · 5 ep · свіжий запуск','+0,4329 в.п.',cols.D]];
   routes.forEach(([lab,val,c],i)=>{const yy=205+i*76;dot(s,690,yy+20,8,c);txt(s,lab,715,yy,375,42,17,i===1?ink:muted,i===1);txt(s,val,1080,yy,105,42,16,c,true,'right');if(i<routes.length-1)ln(s,690,yy+31,690,yy+70,lineColor,1);});
   txt(s,'Δ до власного старту за v2 validation',710,598,450,23,14,muted,false,'right');return;
  }
  const y0=240,row=68,colors=[cols.N,cols.A,cols.B,cols.D];labels.forEach((lab,i)=>{const yy=y0+i*row;dot(s,690,yy+19,8,colors[i%4]);txt(s,lab,715,yy,460,40,18,i===0?ink:muted,i===0);if(i<labels.length-1)ln(s,690,yy+31,690,yy+row-7,lineColor,2);});return;
 }
 if(kind==='bigMetric'){txt(s,'78,40%',cx,230,cw,125,86,cols.N,true,'center');txt(s,'Вихідна pretrained YOLO26n',cx,365,cw,44,24,ink,true,'center');txt(s,'CUB validation · mAP50–95',cx,424,cw,34,18,muted,false,'center');txt(s,'600 зображень',cx,465,cw,30,17,muted,false,'center');return;}
 if(kind==='studyS'){barChart(s,['Raw S','S → head3'],[71.1975,71.4432],{colors:[cols.C,cols.plan],x:90,y:250,w:1070,h:230,max:82,unit:'%'});txt(s,'Δ до власного старту +0,2457 в.п. · N стартувала з 77,2377%',100,540,1080,32,17,muted);return;}
 if(kind==='neckCompare'){barChart(s,['D → head3','D → neck+head3'],[78.4981,78.0153],{colors:[cols.head,cols.plan],x:90,y:250,w:1070,h:230,max:82,unit:'%'});txt(s,'Різниця head-only мінус neck+head: +0,4828 в.п.',100,540,1080,32,17,muted);return;}
 if(kind==='headCompare'){barChart(s,['D · голова · 3 епохи','Окремий запуск · 5 епох'],[78.4981,78.4039],{colors:[cols.head,cols.plan],x:90,y:250,w:1070,h:230,max:82,unit:'%'});txt(s,'Найкращі ваги: після епохи 2 та епохи 1 відповідно',100,540,1080,32,17,muted);return;}
 if(kind==='gtCompare'){txt(s,'Однакові ваги D',cx,250,cw,45,24,ink,true,'center');txt(s,'51,98%',cx,322,230,72,46,'#F8C76B',true,'center');txt(s,'legacy GT',cx,395,230,30,18,muted,false,'center');txt(s,'77,97%',cx+300,322,230,72,46,cols.D,true,'center');txt(s,'v2 GT',cx+300,395,230,30,18,muted,false,'center');txt(s,'Зміна ground truth · не приріст ваг',cx,482,cw,38,19,cols.plan,true,'center');return;}
 // A quiet fallback still carries a labeled, editable evidence graphic.
 const tag=kind==='seeds'?'Критерій: п’ять повторів':kind==='counts'?'Розподіл даних':kind==='gtCompare'?'Незмінні ваги · інше GT':kind.toUpperCase();
 txt(s,tag,cx,300,cw,60,25,cols.D,true,'center');
}

for(let i=0;i<60;i++){
 const [title,claim,kind,narration,source]=data.slides[i],n=i+1,s=P.slides.add();
 s.background.fill=bg;
 const planned=n>=53&&n<=60;
 rect(s,0,0,1280,9,planned?cols.plan:cols.D);
 txt(s,n<=12?'ПРОЄКТ · ДАНІ · МЕТРИКИ':n<=20?'ІСТОРИЧНИЙ BASELINE':n<=42?'V2 · КОРОТКА СЕРІЯ':n<=52?'АУДИТ · СТЕНД':'НАСТУПНИЙ ПРОТОКОЛ',72,25,730,23,14,planned?cols.plan:muted,true);
 txt(s,title,72,58,1135,65,n===1?44:32,ink,true);
 if(n===1){txt(s,claim,72,151,500,140,25,muted);addImageOrChart(s,n,kind);}
 else if(['cubBars','videoBars','deltaCub','deltaVideo','scatterLegacy','scatterV2','daCompare','studyAll','studyDelta','studyStart','studyS','neckCompare','headCompare','photoVideo','tilesAP','tilesFPS','seedResults','benchmark','throughput','memory'].includes(kind)){
  txt(s,claim,80,134,1100,54,20,ink);addImageOrChart(s,n,kind);
 }else{
  txt(s,claim,75,148,500,170,n===20||n===28||n===41?22:22,ink);
  addImageOrChart(s,n,kind);
 }
 txt(s,`ДЖЕРЕЛО: ${source}`,80,632,1080,20,11,muted);
 txt(s,String(n).padStart(2,'0'),1140,665,60,22,14,muted,true,'right');
 txt(s,'BIRD DETECTOR · УКРАЇНСЬКА ВЕРСІЯ',72,665,450,22,13,muted);
 s.speakerNotes.text=`Сценарій диктора:\n${narration}\n\nДжерело: ${source}.\nРедакційна примітка: округлення відображення не змінює точні значення в metrics_manifest.json.`;
 const png=await P.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(renders,`slide-${String(n).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const pptx=await PresentationFile.exportPptx(P);
const stagedPptx=path.join(build,'Bird_Detector_Presentation_UK_without_audio.pptx');
await pptx.save(stagedPptx);
const {execFile}=await import('node:child_process');
const {promisify}=await import('node:util');
await promisify(execFile)('python3',[
 path.join(here,'embed_slide_audio.py'),
 stagedPptx,
 path.join(repo,'audio','slide_clips'),
 path.join(here,'assets','audio-play.png'),
 path.join(output,'Bird_Detector_Presentation_UK.pptx'),
 path.join(build,'audio_mp3'),
],{maxBuffer:1024*1024});
console.log(`Створено ${P.slides.items.length} слайдів, вбудовано окреме озвучення для кожного слайду.`);
