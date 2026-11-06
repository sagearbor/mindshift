"""Plot model matches to human consensus on the identical English screen."""
import json
from pathlib import Path
from .benchmark import sha256, save_json

REPORTS=Path(__file__).parent/'reports'


def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=REPORTS/'native-screen-comparison.json'
    data=json.loads(source.read_text())
    labels=['anger','disgust','fear','happiness','neutral','sadness']
    names=['Our Whisper small (float)','Qwen3-Omni 30B (4-bit)','Liquid LFM3-Audio 2B','EmotionThinker 7B','MERaLiON (training overlap)*']
    assert len(data['entries'])==len(names)==5 and len(data['ids'])==120
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','svg.hashsalt':'human-consensus-v1'})
    fig,ax=plt.subplots(figsize=(13.5,7.7));fig.patch.set_facecolor('white');ax.set_facecolor('white')
    colors=['#185e9b','#7958a4','#a76812','#187c7d','#626973'];markers=['o','s','^','D','X'];coordinates=[]
    for i,(entry,name,color,marker) in enumerate(zip(data['entries'],names,colors,markers)):
        xs=[];ys=[]
        for j,label in enumerate(labels):
            row=entry['metrics']['per_emotion'][label]
            assert row['support']==20
            x=100*row['recall'];y=j+(i-2)*.135;xs.append(x);ys.append(y)
            coordinates.append(dict(model=entry['model'],emotion=label,percent_correct=x,n=20,x=x,y=y))
        ax.scatter(xs,ys,s=80,c=color,marker=marker,label=f'{name}\n{entry["metrics"]["accuracy"]*100:.1f}% overall',zorder=3)
    for j in range(6):ax.axhline(j,color='#e9edf2',linewidth=1,zorder=0)
    ax.axvline(100,color='#177245',linestyle='--',linewidth=1.5)
    ax.set_xlim(-3,104);ax.set_ylim(5.55,-.55);ax.set_xticks(range(0,101,20));ax.set_yticks(range(6),[x.title() for x in labels])
    ax.set_xlabel('Percent matching the human consensus — higher is better',labelpad=15)
    ax.grid(axis='x',color='#e5e8ed');ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#c9d0d8');ax.tick_params(length=0,pad=8)
    ax.legend(loc='upper left',bbox_to_anchor=(1.03,1.03),frameon=False,labelspacing=1.35,fontsize=11,handletextpad=.8)
    fig.text(.04,.95,'How often did each model agree with the human labelers?',fontsize=21,fontweight='bold',color='#17263b')
    fig.text(.04,.90,'Same 120 English recordings • 18 speakers • 20 recordings per emotion • all models run locally',fontsize=12,color='#445469')
    fig.text(.04,.105,'Human reference = the most-selected emotion among audio-only listeners; tied votes excluded.',fontsize=11,color='#344356')
    fig.text(.04,.074,'100% means matching that reference every time. It is not a measured human accuracy score.',fontsize=11,color='#344356')
    fig.text(.04,.043,'* MERaLiON trained on this corpus. All rows are development screening, not fresh test confirmation.',fontsize=10.5,color='#344356')
    fig.subplots_adjust(left=.13,right=.69,top=.84,bottom=.22)
    fig.savefig(REPORTS/'human-agreement.svg',metadata={'Date':None},facecolor='white')
    fig.savefig(REPORTS/'human-agreement.png',dpi=160,facecolor='white')
    save_json(REPORTS/'human-agreement.json',dict(source_sha256=sha256(source),svg_sha256=sha256(REPORTS/'human-agreement.svg'),coordinates=coordinates))
    plt.close(fig)


if __name__=='__main__':build()
