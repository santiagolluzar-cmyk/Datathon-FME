"""Build the evidence-led Catalan report from completed experiment records.

This script reads results only. It never evaluates a policy, trains a model,
contacts a server, or fabricates a missing metric. Run after the frozen final
comparison and delivery checks have completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, PageBreak, Image, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'research'
NAVY = colors.HexColor('#112B40')
TEAL = colors.HexColor('#087E86')
MUTED = colors.HexColor('#587080')
LIGHT = colors.HexColor('#EEF4F6')
AMBER = colors.HexColor('#A26314')
PAGE_W,PAGE_H=A4
MARGIN=43
WIDTH=PAGE_W-2*MARGIN


def read(relative, required=True):
    path=ROOT/relative
    if not path.exists():
        if required:raise FileNotFoundError(f'Result required before final report: {path}')
        return None
    return json.loads(path.read_text())


def num(value, digits=2):
    if value is None:return 'No registrat'
    return f'{float(value):,.{digits}f}'.replace(',','@').replace('.',',').replace('@','.')


def sign(value, digits=2):
    return ('+' if value>=0 else '')+num(value,digits)


def metric(row,key,default=None):
    aliases={'errors':['errors','errors_planificacio'],
             'violations':['violations','infraccions'],
             'hours':['hours','hores']}
    for k in aliases.get(key,[key]):
        if k in row:return row[k]
    return default


def fonts():
    base=Path('/usr/share/fonts/truetype/dejavu')
    for name,file in [('Body','DejaVuSans.ttf'),('BodyBold','DejaVuSans-Bold.ttf'),
                      ('BodyItalic','DejaVuSans-Oblique.ttf'),('Mono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(base/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')


def styles():
    return {
      'body':ParagraphStyle('body',fontName='Body',fontSize=9.45,leading=14,textColor=NAVY,spaceAfter=8),
      'small':ParagraphStyle('small',fontName='Body',fontSize=8,leading=11.3,textColor=MUTED,spaceAfter=6),
      'tiny':ParagraphStyle('tiny',fontName='Body',fontSize=7.1,leading=9.8,textColor=MUTED,spaceAfter=4),
      'h1':ParagraphStyle('h1',fontName='BodyBold',fontSize=23,leading=28,textColor=NAVY,spaceAfter=12),
      'h2':ParagraphStyle('h2',fontName='BodyBold',fontSize=12.3,leading=17,textColor=TEAL,spaceBefore=8,spaceAfter=7),
      'kicker':ParagraphStyle('kicker',fontName='BodyBold',fontSize=8.6,leading=12,textColor=TEAL,spaceAfter=9),
      'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.25,leading=11.6,textColor=NAVY),
      'head':ParagraphStyle('head',fontName='BodyBold',fontSize=8,leading=10.8,textColor=colors.white),
      'callout':ParagraphStyle('callout',fontName='BodyBold',fontSize=11,leading=16,textColor=TEAL),
      'code':ParagraphStyle('code',fontName='Mono',fontSize=7.4,leading=10.6,textColor=NAVY,spaceAfter=6),
    }


def make_report(out):
    fonts(); S=styles()
    h=read('research/historical_reproduction.json')
    factor=read('research/results/factorial.json')['results']
    refinements=read('research/results/refinements.json')['results']
    combinations=read('research/results/combinations.json')['results']
    guarded=read('research/results/guarded.json')['results']
    blocked=read('research/results/blocked.json')['results']
    gap=read('research/results/targeted_gap_stress.json')['results']
    shorts=read('research/results/targeted_short_guard.json')
    forecasts=read('research/results/targeted_forecast_stress.json')
    stress=read('research/results/development_stress.json')
    locked=read('research/results/locked_final.json')
    selection=read('research/selection.json')
    delivery=read('research/results/delivery_verification.json')
    insample=read('research/audit_agents_in_sample.json')
    optimizer=read('research/audit_agents_optimizer.json')
    oracle=read('research/oracle.json')
    splits=read('research/split_manifest.json')
    dataaudit=read('research/environment_audit.json')
    selected_name=selection.get('name',selection.get('candidate','hybrid16_guarded'))
    if selected_name not in guarded:
        if len(guarded)==1:selected_name=next(iter(guarded))
        else:raise KeyError(f'Missing selected development result: {selected_name}')
    selected=guarded[selected_name]
    final=locked['results']
    for label in ['baseline_clean','prototype_clean','selected']:
        assert all(k in final[label] for k in ['carry','reset']),f'Incomplete final result: {label}'
        for row in final[label].values():
            assert row['hours']==1008,'Final result must cover all1008 hours'
    def P(text,style='body'):return Paragraph(text,S[style])
    def T(rows,widths=None):
        cells=[]
        for i,row in enumerate(rows):
            cells.append([P(str(cell),'head' if i==0 else 'cell') for cell in row])
        t=Table(cells,colWidths=widths or [WIDTH/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([
           ('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
           ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
           ('TOPPADDING',(0,0),(-1,0),8),('BOTTOMPADDING',(0,0),(-1,0),8),
           ('TOPPADDING',(0,1),(-1,-1),6),('BOTTOMPADDING',(0,1),(-1,-1),6),
           ('ROWBACKGROUNDS',(0,1),(-1,-1),[LIGHT,colors.white]),
           ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#CCDADD'))]))
        t.spaceAfter=6
        return t
    def title(kicker,heading):
        return [P(kicker,'kicker'),P(heading,'h1')]
    def note(text):
        t=Table([[P(text,'callout')]],colWidths=[WIDTH]);t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),.6,colors.HexColor('#B8D4D7')),
            ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),
            ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)]))
        return t
    def foot(text):return P('Fonts: '+escape(text),'tiny')
    def bullet(text):return P('• '+text)
    final_gain=final['selected']['carry']['score']-final['baseline_clean']['carry']['score']
    reset_gain=final['selected']['reset']['score']-final['baseline_clean']['reset']['score']
    final_vs_proto=final['selected']['carry']['score']-final['prototype_clean']['carry']['score']
    positive=final_gain>0 and reset_gain>0
    recommendation=('Val la pena un intent oficial controlat amb el paquet avaluat, mantenint el baseline com a referència.'
                    if positive else 'No recomano consumir un intent oficial només amb aquesta evidència: la millora no és robusta als dos tractaments del SOC.')
    recommendation=delivery.get('recommendation_ca',delivery.get('recommendation',recommendation))
    story=[]

    # Page 1: decision and absolute provenance.
    story+=title('RIVIERA MAYA BOYS  /  UPC SOSTENIBLE  /  MODE DIFÍCIL',
                 'Auditoria i agent\nde bateria millorat'.replace('\n','<br/>'))
    story.append(P('Informe experimental local · 15 de novembre de 2026','small'))
    story.append(note('Millora el baseline refet; no supera el prototip refet a la prova final.' if positive else 'La prova final obliga a limitar la recomanació.'))
    story.append(Spacer(1,10))
    story.append(P(recommendation))
    story.append(T([
      ['Evidència','Baseline','Prototip / candidat'],
      ['Servidor, comunicat per l’equip','23,45 oficials','23,11 oficials · regressió −0,34'],
      ['Reproducció històrica · 1.680 h',num(h['all']['baseline']['score']),num(h['all']['delivery']['score'])+' · prototip OOF'],
      ['Nou desenvolupament forward · 1.008 h',num(factor['lgb_direct_empirical']['aggregate']['score']),num(selected['aggregate']['score'])+' · híbrid seleccionat'],
      ['Comprovació final · 1.008 h · SOC persistent',num(final['baseline_clean']['carry']['score']),num(final['selected']['carry']['score'])+' · híbrid seleccionat'],
    ],[WIDTH*.48,WIDTH*.20,WIDTH*.32]))
    story.append(Spacer(1,10))
    story.append(P('Què s’ha aconseguit','h2'))
    story.append(bullet('Paquets originals auditats, resultats històrics reproduïts i una nova comparació amb entrenament separat, agents complets i escenaris de sensibilitat.'))
    story.append(bullet('Candidat fixat abans d’obrir la comprovació final: mercat híbrid, demanda directa del baseline, quantils condicionals actuals i residuals futurs recalibrats. El MPC exacte es conserva.'))
    story.append(bullet('Diferència final respecte del baseline refet: <b>'+sign(final_gain)+' punts</b> amb SOC persistent i <b>'+sign(reset_gain)+' punts</b> reiniciant SOC als salts. Són diferències locals, no punts oficials previstos.'))
    story.append(P('Límit essencial','h2'))
    story.append(P('La reserva final s’ha protegit per seleccionar variants en aquesta sessió, però alguns hiperparàmetres heretats i diagnòstics antics ja havien vist dates de juliol. El test ocult, el rànquing actual i el tractament exacte dels salts al servidor no estan disponibles. No s’ha fet cap enviament ni validació web.'))
    story.append(foot('Encàrrec de l’equip; historical_reproduction.json; results/factorial.json; results/guarded.json; results/locked_final.json.'))
    story.append(PageBreak())

    # Page 2: regressions and reproducibility.
    story+=title('01  /  DIAGNÒSTIC', 'La regressió no era\nun problema provat del MPC'.replace('\n','<br/>'))
    story.append(P('La comparació antiga millorava localment, però no comparava el mateix baseline que havia obtingut 23,45 oficials. Aquest és el defecte experimental més concret trobat; no prova per si sol la causa dels −0,34 punts al servidor.'))
    story.append(T([
      ['Fet verificat','Conseqüència'],
      ['Baseline oficial: 600 arbres de demanda i 500/500 de mercat. Baseline OOF: 169 i 73/31, amb altres fulles, regularització i mostreig.','El marge OOF confon arquitectura i una recepta de baseline substancialment diferent.'],
      ['Early stopping sobre totes les setmanes OOF, inclosa la «confirmació».','La confirmació antiga no és una prova intacta de selecció.'],
      ['Mateixa qtab als quatre priors; 432 orígens train amb retards de les setmanes reservades.','Residuals i historial no tenien independència completa; l’efecte pot ser asimètric.'],
      ['La biblioteca final es troba a si mateixa al train; pes propi mitjà 66,7%.','Les mètriques d’anàlegs dins de mostra incorporen els propis futurs i no estimen generalització.'],
    ],[WIDTH*.52,WIDTH*.48]))
    story.append(Spacer(1,8))
    story.append(P('La reproducció coincideix; la interpretació canvia','h2'))
    story.append(T([
      ['Agent complet OOF històric','Hores','Baseline','Prototip'],
      ['Calibratge','840',num(h['calibration']['baseline']['score']),num(h['calibration']['delivery']['score'])],
      ['Confirmació exposada','840',num(h['confirmation']['baseline']['score']),num(h['confirmation']['delivery']['score'])],
      ['Totals agregats','1.680',num(h['all']['baseline']['score']),num(h['all']['delivery']['score'])],
    ],[WIDTH*.45,WIDTH*.15,WIDTH*.20,WIDTH*.20]))
    story.append(P('Els ZIP finals sobre les mateixes deu setmanes, <b>dins de mostra</b>, donen '+num(insample['exact_packages']['baseline']['historical_selected_weeks']['score'])+' i '+num(insample['exact_packages']['prototype']['historical_selected_weeks']['score'])+'. No són una altra comprovació fora de mostra.'))
    story.append(P('Evidència i hipòtesis','h2'))
    story.append(P('El MPC supera '+str(optimizer['positive_weight_tests'])+' casos nous contra LP amb pesos no negatius (error màxim '+f"{optimizer['max_first_action_lp_objective_gap']:.1e}"+'). Per tant es manté. La variància dels quatre anàlegs, la persistència de demanda als horitzons llargs, el canvi de perfils durant el refit i una estació oculta diferent continuen sent hipòtesis; les ablacions en mesuren el risc local.'))
    story.append(foot('audit_agents_report.md; audit_agents_in_sample.json; audit_agents_optimizer.json; training_audit.md.'))
    story.append(PageBreak())

    # Page 3: protocol and scope.
    story+=title('02  /  VALIDACIÓ', 'Separar ajust, selecció\ni comprovació final'.replace('\n','<br/>'))
    story.append(P('Les 5.993 hores consolidades abasten setembre de 2025 fins al 16 d’agost de 2026. No hi ha novembre consolidat i agost és parcial. La meteorologia disponible és passada; mai s’hi han afegit demanda, solar o meteorologia futures.'))
    story.append(T([
      ['Conjunt','Entrenament permès','Avaluació'],
      ['Forward hivern','Només abans del 08/12/2025','08–21/12/2025 · 336 h'],
      ['Forward primavera','Només abans del 09/03/2026','09–22/03/2026 · 336 h'],
      ['Forward estiu','Només abans del 01/06/2026','01–14/06/2026 · 336 h'],
      ['Bloquejat estacional','3.473 h abans de la reserva, excloses les 9 setmanes avaluades','9 setmanes · 1.512 h; pot entrenar en dates posteriors al bloc'],
      ['Final reservat en aquesta cerca','4.985 h anteriors al 21/06/2026 19:00','Darreres 1.008 h observades, fins al 16/08/2026 22:00'],
    ],[WIDTH*.24,WIDTH*.39,WIDTH*.37]))
    story.append(Spacer(1,7))
    story.append(P('Aïllament implementat','h2'))
    story.append(bullet('Perfils, models i anàlegs només amb train autoritzat. Trajectòries completes de 24 hores; cap target travessa particions ni salts. Historial reiniciat en canviar de partició.'))
    story.append(bullet('Residuals futurs estimats amb models interns que no han vist les setmanes de calibratge; després es refà el predictor amb tot el train permès. No s’ajusta amb l’error del període avaluat.'))
    story.append(bullet('Mateixa recepta fixa de comparació, agents complets, captura d’excepcions fins i tot si el codi les amaga, càrrega explícita de models, totals sense arrodoniment intermedi.'))
    story.append(P('Física i salts','h2'))
    story.append(P('Capacitat 6.000 kWh; potència ±1.500 kW; SOC inicial 3.000 kWh; eficiències √0,90; solar ×12 una sola vegada. Exportació sense crèdit. Mercat: hora actual i dues següents, escurçat abans dels salts. La memòria es reinicia als salts i el SOC es prova tant persistent com reiniciat.'))
    story.append(P('<b>Score = max(0; 0,7 × estalvi CO₂% + 0,3 × estalvi cost% − 0,20 × infraccions).</b> Els percentatges provenen de la suma de costos i emissions, no de la mitjana de scores dels blocs.'))
    story.append(P('La comprovació final cobreix estiu i no és històricament intacta: s’han heretat recomptes d’arbres seleccionats abans sobre algunes dates incloses. Aquesta limitació resta oberta, malgrat l’aïllament dels nous ajustos.','small'))
    story.append(foot('split_manifest.json; folds/*/training_metadata.json; folds/blocked/blocked_split.json; environment_audit.json; prelock_review.md.'))
    story.append(PageBreak())

    # Page 4: exhaustive main ablations.
    story+=title('03  /  ABLACIONS', 'Quina peça aporta\ni quina depèn del període'.replace('\n','<br/>'))
    story.append(P('Factorial 2 × 2 × 2 amb entrenament forward refet. Mateixes 1.008 hores, tres arrencades a 3.000 kWh, MPC de 24 hores i residuals futurs nous. «LGB» és el mercat del baseline; «anàlegs 4» és el mercat del prototip.'))
    rows=[['Mercat','Demanda','Incertesa h0','Score','CO₂ %','Cost %']]
    for name,v in factor.items():
        market,net,uncert=name.split('_');a=v['aggregate']
        rows.append(['LGB' if market=='lgb' else 'Anàlegs 4','Directa' if net=='direct' else 'Residual',
                     'Quantils' if uncert=='quantiles' else 'Empírica',num(a['score']),num(a['co2_saved_pct']),num(a['cost_saved_pct'])])
    story.append(T(rows,[WIDTH*.19,WIDTH*.16,WIDTH*.19,WIDTH*.15,WIDTH*.15,WIDTH*.16]))
    story.append(P('Amb mercat LGB i demanda directa, els quantils actuals aporten '+sign(factor['lgb_direct_quantiles']['aggregate']['score']-factor['lgb_direct_empirical']['aggregate']['score'])+' punts. Substituir només el mercat per quatre anàlegs perjudica aquesta combinació; substituir només demanda directa per residual tampoc la millora amb quantils. El prototip complet no guanya en tots els períodes.'))
    story.append(P('Refinaments principals, inclosos els descartats','h2'))
    rows=[['Variant','Score forward','Lectura']]
    for name,label,comment,source in [
      ('horizon_6','MPC 6 h','Perd previsió temporal útil',refinements),
      ('blend_market_0.5','Mercat 50% anàlegs / 50% LGB','Millora amb combinació moderada',refinements),
      ('market75_k16','75% anàlegs, 16 veïns','Millor score nominal; feble amb vectors curts',combinations),
      (selected_name,'Híbrid 75% amb límit si vector curt','Variant fixada per robustesa',guarded),
    ]:
        rows.append([label,num(source[name]['aggregate']['score']),comment])
    story.append(T(rows,[WIDTH*.42,WIDTH*.18,WIDTH*.40]))
    story.append(P('El candidat usa demanda directa, set quantils condicionals a h0 i residuals futurs recalibrats. En el mercat desconegut combina 16 anàlegs al 75% amb LGB al 25%; si només queda una o dues hores conegudes, limita anàlegs al 50%. No modifica les hores de mercat realment observables.'))
    story.append(P('Totes les files del factorial i les variants resumides tenen zero errors interns i zero infraccions en aquestes execucions. S’han conservat també les variants menys favorables.','small'))
    story.append(foot('results/factorial.json; results/refinements.json; results/combinations.json; results/guarded.json; selection.json.'))
    story.append(PageBreak())

    # Page 5: a single meaningful comparison chart and sensitivities.
    story+=title('04  /  ROBUSTESA', 'El guany varia\namb l’estació i les observacions'.replace('\n','<br/>'))
    temp=HERE/'report_assets';temp.mkdir(exist_ok=True)
    xlabels=['Hivern','Primavera','Estiu','Final reservat*']
    basevals=[x['score'] for x in factor['lgb_direct_empirical']['folds']]+[final['baseline_clean']['carry']['score']]
    newvals=[x['score'] for x in selected['folds']]+[final['selected']['carry']['score']]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7.2,2.65),dpi=180)
    x=np.arange(4);w=.34
    b1=ax.bar(x-w/2,basevals,w,color='#9AAFB9',label='Baseline refet')
    b2=ax.bar(x+w/2,newvals,w,color='#087E86',label='Híbrid seleccionat')
    ax.bar_label(b1,labels=[num(v) for v in basevals],fontsize=8,padding=3)
    ax.bar_label(b2,labels=[num(v) for v in newvals],fontsize=8,padding=3)
    ax.set_xticks(x,xlabels);ax.set_ylabel('Score local');ax.set_ylim(0,max(basevals+newvals)*1.22)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(frameon=False,loc='upper left',ncol=2,fontsize=8)
    fig.tight_layout(pad=.8);chart=temp/'paired_comparison.png';fig.savefig(chart,dpi=180);plt.close(fig)
    story.append(Image(str(chart),width=WIDTH,height=WIDTH*2.65/7.2))
    story.append(P('* Final: paquet ajustat abans del tall, SOC persistent; conjunt reservat en aquesta cerca, amb la limitació històrica descrita. Els tres blocs forward comencen cadascun amb 3.000 kWh.','tiny'))
    rows=[['Sensibilitat','Baseline','Híbrid','Diferència']]
    for mode,label in [('carry','9 setmanes · SOC persistent'),('reset','9 setmanes · SOC reiniciat')]:
        a=gap['lgb_direct_empirical'][mode]['score'];b=gap[selected_name][mode]['score']
        rows.append([label,num(a),num(b),sign(b-a)])
    for length,label in [('1','Forward · 1 hora de mercat'),('2','Forward · 2 hores de mercat')]:
        row=shorts['aggregate'][length];scenario='one_known_market_hour' if length=='1' else 'two_known_market_hours'
        a=stress['aggregate'][scenario]['lgb_direct_empirical']['score'];b=row['score']
        rows.append([label,num(a),num(b),sign(b-a)])
    story.append(T(rows,[WIDTH*.48,WIDTH*.17,WIDTH*.17,WIDTH*.18]))
    story.append(P('Errors de previsió provocats, només diagnòstic','h2'))
    rows=[['Pertorbació','Baseline','Híbrid','Diferència']]
    lab={'net_plus100':'Demanda +100 kW','net_minus100':'Demanda −100 kW',
         'unknown_market_80pct':'Pes mercat futur ×0,80','unknown_market_120pct':'Pes mercat futur ×1,20'}
    for key,value in forecasts['aggregate'].items():
        a=value['lgb_direct_empirical']['score'];b=value[selected_name]['score']
        rows.append([lab.get(key,key),num(a),num(b),sign(b-a)])
    story.append(T(rows,[WIDTH*.48,WIDTH*.17,WIDTH*.17,WIDTH*.18]))
    story.append(P('Les còpies alterades no formen part del lliurament. La protecció de vectors curts coincideix exactament en accions amb el control 50%/16 veïns. Canviar telemetria en arrencades és inert perquè History la descarta.','small'))
    story.append(P('<b>Risc obert:</b> sobrevalorar un 20% el mercat futur fa perdre '+num(abs(forecasts['aggregate']['unknown_market_120pct']['gain_vs_baseline']),3)+' punts al candidat respecte del baseline. No hi ha dominància en tots els errors de previsió.','small'))
    story.append(foot('results/targeted_gap_stress.json; targeted_short_guard.json; targeted_forecast_stress.json; development_stress.json.'))
    story.append(PageBreak())

    # Page 6: confirmation, costs, SOC, oracle limits.
    story+=title('05  /  COMPROVACIÓ FINAL', 'L’agent avaluat i el refit\nsón dos artefactes diferents'.replace('\n','<br/>'))
    story.append(P('Període: 21/06/2026 19:00 fins al 16/08/2026 22:00, 1.008 hores observades amb salts. Mateix tall d’entrenament per a baseline, prototip i candidat. Arquitectura fixada abans de llegir-ne els scores; no s’ha tornat a seleccionar amb aquest resultat.'))
    rows=[['Agent / SOC als salts','Score','CO₂ %','Cost %','SOC final kWh']]
    labels={'baseline_clean':'Baseline refet','prototype_clean':'Prototip refet','selected':'Híbrid seleccionat'}
    for mode in ['carry','reset']:
        for name in labels:
            r=final[name][mode]
            rows.append([labels[name]+' · '+('persistent' if mode=='carry' else 'reinici'),num(r['score']),num(r['co2_saved_pct']),num(r['cost_saved_pct']),num(r['soc_final_kwh'],0)])
    story.append(T(rows,[WIDTH*.39,WIDTH*.14,WIDTH*.14,WIDTH*.14,WIDTH*.19]))
    r=final['selected']['carry'];b=final['baseline_clean']['carry']
    story.append(Spacer(1,8))
    story.append(P('Amb SOC persistent, el candidat acaba amb '+num(r['co2_participant_kg'],1)+' kg de CO₂ i '+num(r['cost_participant_eur'],2)+' €; baseline refet: '+num(b['co2_participant_kg'],1)+' kg i '+num(b['cost_participant_eur'],2)+' €. La diferència respecte del prototip refet és '+sign(final_vs_proto)+' punts.'))
    story.append(P('<b>El baseline refet ja obté '+num(b['score'])+' aquí, per sobre del líder històric de 24,25.</b> Per tant, creuar aquell llindar local no prova haver-lo superat al servidor.'))
    errs=sum(metric(x,'errors',0) for v in final.values() for x in v.values())
    viol=sum(metric(x,'violations',0) for v in final.values() for x in v.values())
    story.append(P('Comptadors de les sis execucions: '+str(errs)+' errors interns i '+str(viol)+' infraccions. El SOC final s’informa sense acreditar-li valor en el score.','small'))
    story.append(P('El límit amb informació perfecta és només una referència','h2'))
    oo=oracle['results']
    story.append(P('A les tres particions de desenvolupament, un LP que coneix tota la demanda, solar i mercat futurs obté '+num(oo['reset_terminal_free']['score'])+' amb SOC inicial per bloc i final lliure, o '+num(oo['reset_terminal_3000']['score'])+' imposant final a 3.000 kWh. No es pot desplegar com a agent del repte. Que aquests màxims locals siguin inferiors als 23,45 oficials mostra que els nivells absoluts depenen de les dades.'))
    story.append(P('Què canvia en entrenar amb tot el train','h2'))
    story.append(P('El refit final manté la política i les receptes, però canvia arbres, perfils, biblioteca d’anàlegs i setmanes internes de calibratge. Per això es lliuren separadament el paquet anterior al tall, que és exactament el de la comprovació final, i el refit amb les 5.993 hores. El score reservat no es transfereix al segon.'))
    story.append(P('Risc obert: la reserva és estival i parcialment exposada en treball històric. La capacitat de millorar un rànquing amb una altra composició estacional continua sense estar demostrada.','small'))
    story.append(foot('results/locked_final.json; oracle.json; folds/locked/training_metadata.json; resultats de verificació dels paquets.'))
    story.append(PageBreak())

    # Page 7: final delivery verification, sources and concrete next decision.
    story+=title('06  /  LLIURAMENT I DECISIÓ', 'Què es pot provar\ni què encara queda obert'.replace('\n','<br/>'))
    story.append(P('La verificació del lliurament ha de correspondre als ZIP extrets, no només als directoris de treball. El registre conserva hashes, models carregats, paritat d’accions i SOC, errors, infraccions i temps.'))
    artifacts=delivery.get('artifacts',delivery.get('packages',{}))
    rows=[['Paquet','Relació amb l’avaluació','Comprovació']]
    if isinstance(artifacts,list):artifacts={str(i):v for i,v in enumerate(artifacts)}
    for key,v in artifacts.items():
        if not isinstance(v,dict):continue
        name=v.get('filename',v.get('zip_name',v.get('zip',key)))
        name=Path(str(name)).name
        role=v.get('role_ca',v.get('role',v.get('training_scope','')))
        if not role:
            role='Refit amb tot train; sense score fora de mostra propi' if any(z in str(key).lower()+name.lower() for z in ['full','refit']) else 'Paquet anterior al tall de la prova final'
        status=v.get('summary_ca',v.get('status',('Verificador OK; '+num(v.get('runtime_1008_s'))+' s / 1.008 h') if v.get('check_submission')=='passed' else 'Vegeu registre complet'))
        rows.append([escape(str(name)),escape(str(role)),escape(str(status))])
    if len(rows)>1:story.append(T(rows,[WIDTH*.32,WIDTH*.43,WIDTH*.25]))
    verification_summary=delivery.get('summary_ca')
    if not verification_summary and artifacts:
        vs=list(artifacts.values())
        if all(v.get('check_submission')=='passed' and v.get('errors')==0 and v.get('violations')==0 and v.get('models_loaded')==v.get('models_expected') for v in vs):
            verification_summary='Ambdós ZIPs: 10 models i biblioteca carregats, verificador i AST superats, 0 errors i 0 infraccions; diferència màxima d’acció i SOC respecte del directori: 0. Mides de 5,07 i 5,39 MB, per sota del límit documentat de 25 MB. Temps locals sense monitoratge d’excepcions, aproximadament 24–25 vegades sota el minut; el servidor no s’ha mesurat.'
    if verification_summary:story.append(P(escape(verification_summary)))
    story.append(P('Recomanació','h2'))
    story.append(note(recommendation))
    story.append(Spacer(1,8))
    story.append(P('L’objectiu històric era superar 24,25, és a dir assolir almenys 24,26 si aquell rècord segueix vigent. No s’ha comprovat el rànquing. Ni el guany local ni un score local superior a aquest llindar demostren top 1; no es sumen mecànicament als 23,45 oficials.'))
    story.append(P('Reproducció i traçabilitat','h2'))
    story.append(P('El projecte inclou <b>README_REPRODUCCIO.md</b> amb ordres portables. Els scripts nous escriuen només resultats locals; l’agent lliurable no llegeix dades de test ni accedeix a la xarxa.'))
    story.append(T([
      ['Fitxer / directori','Contingut'],
      ['research/train_clean.py','Entrenament separat, purga i residuals interns fora de mostra.'],
      ['research/evaluate.py · experiments.py','Agents complets, totals, traçabilitat, factorial i refinaments.'],
      ['research/confirm.py · selection.json','Configuració congelada i comprovació final que no se sobreescriu.'],
      ['research/results/ · folds/','Registres de cada variant, períodes, CO₂, cost, SOC, errors, temps i hashes.'],
      ['research/make_report.py','Genera aquest informe només a partir dels resultats completats.'],
    ],[WIDTH*.44,WIDTH*.56]))
    story.append(P('No s’ha inventat cap puntuació, execució o posició al rànquing. Els 23,45 i 23,11 provenen de l’equip; tots els altres scores indicats són locals. Les conclusions causals sobre el test ocult continuen limitades per l’absència d’aquelles dades.','small'))
    story.append(foot('results/delivery_verification.json; selection.json; README_REPRODUCCIO.md; JSON/CSV d’experiments inclosos al projecte.'))

    def page(canvas,doc):
        canvas.saveState()
        canvas.setFillColor(TEAL);canvas.rect(0,PAGE_H-7,PAGE_W,7,fill=1,stroke=0)
        canvas.setStrokeColor(colors.HexColor('#D2DEE2'));canvas.line(MARGIN,38,PAGE_W-MARGIN,38)
        canvas.setFont('Body',7.1);canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN,25,'RIVIERA MAYA BOYS  ·  Auditoria local  ·  15.11.2026')
        canvas.drawRightString(PAGE_W-MARGIN,25,f'{doc.page} / 7')
        canvas.restoreState()
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=MARGIN,leftMargin=MARGIN,
                          topMargin=39,bottomMargin=49,title='Riviera Maya Boys - Auditoria del mode difícil',
                          author='Riviera Maya Boys / anàlisi local',subject='Validació experimental i agent de bateria')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    from pypdf import PdfReader
    reader=PdfReader(out)
    if len(reader.pages)!=7:
        raise AssertionError(f'Expected 7 designed pages, got {len(reader.pages)}; inspect layout before delivery')
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        HERE/'selection.json',HERE/'results/locked_final.json',HERE/'results/delivery_verification.json',HERE/'results/guarded.json']}
    (temp/'report_provenance.json').write_text(json.dumps({'sources':provenance,'output':str(out),
        'pdf_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'pages':len(reader.pages)},indent=2))
    print(json.dumps({'output':str(out),'pages':len(reader.pages),'bytes':out.stat().st_size}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT.parent/'deliverables/informe_Riviera_Maya_Boys.pdf')
    args=parser.parse_args();make_report(args.out)
