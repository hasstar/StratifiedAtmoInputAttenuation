from pathlib import Path
import hashlib, json, zipfile, xml.etree.ElementTree as E
import pypdfium2 as pdfium
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ppt=P/"Fig02_Stratified_Atmosphere_Editable.pptx"
pdf=P/"Fig02_Stratified_Atmosphere_Editable.pdf"
ns={"p":"http://schemas.openxmlformats.org/presentationml/2006/main","a":"http://schemas.openxmlformats.org/drawingml/2006/main"}
with zipfile.ZipFile(ppt) as z:
    s=E.fromstring(z.read("ppt/slides/slide1.xml"));sp=s.findall(".//p:sp",ns);cx=s.findall(".//p:cxnSp",ns);pics=s.findall(".//p:pic",ns)
    textshapes=[x for x in sp if ''.join(t.text or '' for t in x.findall('p:txBody//a:t',ns)).strip()]
    names=[x.attrib.get("name") for x in s.findall(".//p:cNvPr",ns)]
    assert len(textshapes)==37 and len(pics)==3 and len(sp)+len(cx)+len(pics)==115
d=pdfium.PdfDocument(str(pdf));assert len(d)==1
pdftext=d[0].get_textpage().get_text_range()
for needle in ["50 base features","106 pressure-level","156 features","Local bilinear interpolation"]:
    assert needle in pdftext,needle
d[0].render(scale=1.6).to_pil().save(P/"QA_PDF_Preview.png")
m=json.loads((P/"source_run/pages/page_001/manifest.json").read_text("utf-8"))
assert len([x for x in m["shapes"] if x["id"].startswith("sample_") and x['type']=='ellipse'])==10
assert len([x for x in m["shapes"] if x["id"].startswith("pressure_")])==10
qa=json.loads((P/"source_run/final/image2ppt_qa.json").read_text("utf-8"))
assert qa["passed"]
r={"passed":True,"pptx_sha256":sha(ppt),"pdf_sha256":sha(pdf),"pdf_pages":len(d),
   "pptx_objects":len(sp)+len(cx)+len(pics),"native_text_objects":len(textshapes),
   "native_non_text_objects":len(sp)+len(cx)-len(textshapes),"independent_bitmap_assets":len(pics),
   "pressure_levels":10,"path_sampling_points":10,"final_image2ppt_qa":True,
   "evidence":"source_run/final/image2ppt_qa.json","pdf_text_verified":True}
(P/"QA_Delivery.json").write_text(json.dumps(r,indent=2)+"\n",encoding="utf-8")
print(json.dumps(r))

