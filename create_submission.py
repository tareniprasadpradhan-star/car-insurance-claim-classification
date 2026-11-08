"""Create the required PDF with two real links first, followed by the entire README."""
from pathlib import Path
import argparse, re
from urllib.parse import urlparse
from html import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab

ROOT=Path(__file__).resolve().parent

def valid_url(value):
    p=urlparse(value)
    if p.scheme!='https' or not p.netloc or any(t in value.upper() for t in ['YOUR_', 'USERNAME', 'YOURREPO']):
        raise argparse.ArgumentTypeError('Supply a real HTTPS URL, not an example placeholder.')
    return value

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--github',type=valid_url)
    parser.add_argument('--app',type=valid_url)
    parser.add_argument('--draft',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.draft and (not args.github or not args.app):
        parser.error('Supply both --github and --app, or use --draft for an unpublished draft.')
    text=(ROOT/'README.md').read_text()
    if not args.draft:
        text=re.sub(r'(## c\. Github Repository Link\n).*?(\n\n)',lambda m:m[1]+args.github+m[2],text,count=1,flags=re.S)
        text=re.sub(r'Live Streamlit App Link: .*',lambda m:'Live Streamlit App Link: '+args.app,text,count=1)
        text=text.replace('The accompanying PDF is a draft until both working links are supplied. No public repository or live app is claimed in this package.',
                          'The repository and deployed app links are supplied above. Verify that both open for the evaluator before submission.')
        (ROOT/'README.md').write_text(text)
    font_dir=Path(reportlab.__file__).parent/'fonts'
    for name,filename in [('Vera','Vera.ttf'),('Vera-Bold','VeraBd.ttf'),('Vera-Italic','VeraIt.ttf'),('Vera-BoldItalic','VeraBI.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(font_dir/filename)))
    pdfmetrics.registerFontFamily('Vera',normal='Vera',bold='Vera-Bold',italic='Vera-Italic',boldItalic='Vera-BoldItalic')
    styles=getSampleStyleSheet()
    for key in ['Title','Heading1','Heading2','Heading3']:
        styles[key].textColor=colors.black
        styles[key].fontName='Vera-Bold'
        styles[key].keepWithNext=True
    styles['Normal'].fontName='Vera';styles['Normal'].fontSize=9;styles['Normal'].leading=13
    styles.add(ParagraphStyle(name='SmallCell',fontName='Vera',fontSize=7.5,leading=10))
    styles.add(ParagraphStyle(name='CodeWrap',fontName='Vera',fontSize=7.5,leading=10,spaceAfter=4,wordWrap='CJK'))
    output=args.output or ROOT/('Car_Insurance_Assignment_Draft.pdf' if args.draft else 'Car_Insurance_Assignment_Submission.pdf')
    output.parent.mkdir(parents=True,exist_ok=True)
    story=[Paragraph('Car Insurance Claim Classification',styles['Title']),Paragraph('Machine learning assignment',styles['Heading2'])]
    for label,link in [('1. GitHub Repository Link',args.github),('2. Live Streamlit App Link',args.app)]:
        story.append(Paragraph(label,styles['Heading2']))
        story.append(Paragraph(f'<link href="{escape(link,quote=True)}" color="#126485">{escape(link)}</link>' if link else 'Pending publication - not yet available',styles['Normal']))
    if args.draft:
        story.append(Spacer(1,10));story.append(Paragraph('DRAFT - Complete the upload and deployment steps, then regenerate this PDF with your actual working links before submitting.',styles['Normal']))
    story += [Spacer(1,16),Paragraph('Complete GitHub README content',styles['Heading1'])]
    lines=text.splitlines();i=0;code=False
    while i<len(lines):
        line=lines[i]
        if line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                block.append(Paragraph(escape(lines[i]).replace(' ','&#160;') or '&#160;',styles['CodeWrap']));i+=1
            story.append(KeepTogether(block));i+=1;continue
        if code:
            story.append(Paragraph(escape(line).replace(' ','&#160;') or '&#160;',styles['CodeWrap']));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',c) for c in row): rows.append(row)
                i+=1
            count=len(rows[0]);widths=([133]+[61]*6) if count==7 else [499/count]*count
            cells=[[Paragraph(escape(c),styles['SmallCell']) for c in row] for row in rows]
            t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DCEAF0')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
            if story and isinstance(story[-1],Paragraph) and story[-1].style.name.startswith('Heading'):
                heading_item=story.pop();story.append(KeepTogether([heading_item,t,Spacer(1,10)]))
            else: story += [t,Spacer(1,10)]
            continue
        if not line.strip():story.append(Spacer(1,5));i+=1;continue
        heading=re.match(r'^(#{1,3}) (.*)',line)
        if heading:story.append(Paragraph(escape(heading[2]),styles[{1:'Heading1',2:'Heading2',3:'Heading3'}[len(heading[1])]]))
        else:
            s=escape(line)
            s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
            story.append(KeepTogether([Paragraph(s,styles['Normal'])]) if s.startswith('<b>') else Paragraph(s,styles['Normal']))
        i+=1
    def footer(c,doc):
        c.setFont('Vera',8);c.setFillColor(colors.HexColor('#596A72'));c.drawString(48,26,'Car Insurance Claim Classification');c.drawRightString(A4[0]-48,26,str(doc.page))
    SimpleDocTemplate(str(output),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=43).build(story,onFirstPage=footer,onLaterPages=footer)
    print(output)
if __name__=='__main__':main()
