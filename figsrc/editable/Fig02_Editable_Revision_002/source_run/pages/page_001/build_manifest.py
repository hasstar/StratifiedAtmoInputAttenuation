"""Author the authoritative page manifest. Does not change experiment data.
All positions are measured in the source image's 1634 x 962 pixel coordinate system.
The Image2PPT CLI is the sole PPTX builder; this file just writes its JSON input.
"""
from pathlib import Path
import json, math
P=Path(__file__).resolve().parent
req=json.loads((P/"page_request.json").read_text("utf-8"))
navy="#071B42"; blue="#07377A"; orange="#F68013"
m={"schema_version":2,"slide":req["slide"],"content_box":req["content_box"],
   "source":{"path":"source.png","width_px":1634,"height_px":962},
   "page_strategy":"mixed-reconstruction with native scientific geometry and three bounded separated illustrations",
   "text_inventory":[],"visual_inventory":[],
   "background_strategy":{"mode":"native-or-script","source_consistency_contract":"White background; measured three-panel composition; ten curved pressure surfaces; source positions and palette retained.",
    "removed_foreground":"Satellite, receiver and Earth supplied as three separate transparent assets; all labels are native text.",
    "comparison_note":"White canvas and three panel boundaries preserve the source layout. Pressure-band fades use editable light solid strips."},
   "quality_checks":{k:True for k in ["font_size_calibrated","visual_inventory_matched","background_strategy_checked","shape_corner_geometry_checked"]},
   "quality_evidence":{
    "font_size_calibrated":{"observation":"Source Latin labels measured visually at 24 px, headers 31 px and panel letters 34 px; consistent levels use 18, 23.25 and 25.5 pt."},
    "visual_inventory_matched":{"observation":"Three separated illustrations, ten pressure curves with ten dots, four grid corner nodes and two feature cards inventoried."},
    "background_strategy_checked":{"observation":"Scientific structure is drawn on white native background; each pictorial asset is bounded to its own source region."},
    "shape_corner_geometry_checked":{"observation":"Inset grid has square corners; feature cards use measured 14 px radii and the feature total card uses a 10 px radius."}},
   "text_boxes":[],"shapes":[],"images":[],"asset_provenance":[],
   "formula_inventory":[],
   "warnings":["Minor texture and edge differences in image-tool-separated illustrations; smooth pressure-band fades reconstructed as light editable strips.",
               "No OCR text lines detected; text transcribed and checked against source image. This is a scientific schematic, not a data plot."]}
regions=[]
for rid,label,box,strategy in [
("a","Propagation through pressure levels",[0,0,1090,962],"mixed-reconstruction"),
("b","Local bilinear interpolation",[1060,0,574,520],"native-object-decomposition"),
("c","Retained meteorological inputs",[1025,528,609,434],"native-object-decomposition")]:
    regions.append({"id":rid,"label":label,"source_bbox_px":box,"risk_level":"high","strategy":strategy,
      "reason":"Measured native geometry retains editable scientific structure; pictorial objects remain independent where applicable.",
      "manifest_ids":{"shapes":[],"text_boxes":[],"images":[]},"protected_visual_anchors":[]})
def add(kind,obj,r):
    m[kind].append(obj); regions[r]["manifest_ids"][kind].append(obj["id"])
    if kind=="text_boxes": m["text_inventory"].append(obj["text"])
    if kind=="shapes":
        ak = "circle-node" if obj["type"]=="ellipse" else ("directed-connector" if obj.get("end_arrow") else "connector") if obj["type"]=="line" else "smooth-curve" if obj["type"]=="bezier" else "card-edge"
        anchor={"id":"anchor-"+obj["id"],"kind":ak,"manifest_id":obj["id"]}
        anchor["source_points_px" if obj["type"]=="line" else "source_bbox_px"]=obj.get("points_px",obj.get("box_px"))
        regions[r]["protected_visual_anchors"].append(anchor)
    return obj
def text(i,t,box,r=0,fs=18,bold=False,color=navy,align="left",rotation=None):
    o={"id":i,"text":t,"box_px":box,"font":"Times New Roman","font_size":fs,"font_size_source":"measured",
       "fit_text":False,"color":color,"bold":bold,"align":align,"valign":"middle","z_index":60}
    if rotation: o["rotation"]=rotation
    return add("text_boxes",o,r)
def line(i,pts,r=0,col=navy,width=1,arrow=False,dash=None,z=25):
    o={"id":i,"type":"line","points_px":pts,"stroke":col,"stroke_width":width,"z_index":z}
    if arrow:o.update(connector="straight",start_arrow="none",end_arrow="triangle",arrow_width="med",arrow_length="med")
    if dash:o["dash"]=dash
    return add("shapes",o,r)
def shape(i,typ,box,r=0,fill="none",stroke=navy,width=1,z=10,**kw):
    return add("shapes",dict(id=i,type=typ,box_px=box,fill=fill,stroke=stroke,stroke_width=width,z_index=z,**kw),r)
def bez(i,segments,r=0,col=navy,width=1,z=15):
    xs=[s[k] for s in segments for k in [0,2,4,6]];ys=[s[k] for s in segments for k in [1,3,5,7]]
    return shape(i,"bezier",[min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys)],r,stroke=col,width=width,z=z,bezier_px=segments)
def mix(c,frac):
    v=[int(c[j:j+2],16) for j in (1,3,5)]
    return "#"+"".join(f"{round(x*frac+255*(1-frac)):02X}" for x in v)
def pic(i,box,r):
    add("images",{"id":i,"path":"assets/"+i+".png","box_px":box,"alt":i+" source-faithful separated illustration","z_index":40},r)
    m["asset_provenance"].append({"path":"assets/"+i+".png","source":"assets/pictorial-sheet.png","source_type":"asset-sheet-separated",
    "provenance_note":"Image-gen separated the source illustration into a transparent sheet; Image2PPT process-sheet split its alpha components. Prompt and job ledger retained."})
    m["visual_inventory"].append({"id":i,"kind":"foreground-asset","representation":"asset-sheet-separated","path":"assets/"+i+".png"})
# Native pressure layers. Three measured anchors determine each quadratic contour.
levels=[1,5,30,200,300,500,650,700,850,925]
left_y=[235,290,344,396,446,499,547,592,635,676]
dot_y=[290,333,376,421,463,506,548,588,626,664]
ends=[(1024,424),(1012,477),(999,529),(990,580),(976,632),(966,684),(953,731),(944,777),(935,815),(925,846)]
cols=["#76B7F1","#6FAFF0","#66BCDF","#4CB2CE","#50ADB7","#69AF91","#99B47C","#CBAE72","#E2A76D","#EFA065"]
dot_x=[]
for k,(lev,y0,yd,(xe,ye),col) in enumerate(zip(levels,left_y,dot_y,ends,cols)):
    x0=70+k*1.8
    xd=835+(389-835)*(yd-120)/(742-120);dot_x.append(xd)
    u=xd-x0; v=xe-x0
    a=((ye-y0)/v-(yd-y0)/u)/(v-u); b=(yd-y0)/u-a*u
    def yy(x):return y0+b*(x-x0)+a*(x-x0)**2
    d=xe-x0
    seg=[x0,y0,x0+d/3,y0+b*d/3,xe-d/3,ye-(b+2*a*d)*d/3,xe,ye]
    # Light, editable bands under each pressure contour; no full-region bitmap.
    for band,(lo,hi,frac) in enumerate([(0,13,.22),(13,26,.13),(26,39,.05)]):
        pts=[[x0+d*j/64,yy(x0+d*j/64)+lo] for j in range(65)]
        pts += [[x0+d*j/64,yy(x0+d*j/64)+hi] for j in reversed(range(65))]
        xs=[v[0] for v in pts];ys=[v[1] for v in pts]
        shape(f"layer_{lev}_shade_{band}","freeform",[min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys)],
              fill=mix(col,frac),stroke="none",z=5,polygon_px=pts)
    bez(f"pressure_{lev}",[seg],col=col,width=1.05,z=12)
    text(f"pressure_label_{lev}",str(lev),[25,y0-14,44,29],align="center")
# Slant line and ten measured dots, with native arrowheads.
line("propagation_path",[835,120,389,742],col=blue,width=2.6,arrow=True,z=26)
line("path_direction",[800,168,762,222],col=blue,width=2.6,arrow=True,z=26)
shape("distance_break_mask","rect",[783,171,29,11],fill="#FFFFFF",stroke="none",z=27)
line("distance_break_1",[787,173,810,173],col=blue,width=1.15,z=28)
line("distance_break_2",[783,180,806,180],col=blue,width=1.15,z=28)
for lev,xd,yd in zip(levels,dot_x,dot_y):
    shape(f"sample_{lev}","ellipse",[xd-6.5,yd-6.5,13,13],fill=orange,stroke=blue,width=.85,z=35)
# Dashed local magnification callout, direction toward the sampled level.
line("local_sampling_callout",[1124,84,dot_x[4]+11,dot_y[4]-1],col="#EF7315",width=1,arrow=True,dash="dash",z=23)
pic("earth",[0,766,1090,196],0)
pic("receiver",[319,729,100,90],0)
pic("satellite",[703,7,303,142],0)
text("panel_a","(a)",[17,7,60,43],fs=25.5,bold=True)
text("pressure_title","Pressure (hPa)",[31,164,210,37],fs=18.5)
text("satellite_label","GEO satellite",[862,17,178,38],fs=19.5)
text("path_label","Slant propagation path",[466,186,310,38],fs=19,bold=True,color=blue)
line("path_label_leader",[688,226,722,263],col=blue,width=1.1,arrow=True,z=25)
text("sample_label","Path sampling points",[589,544,258,39],fs=18.5,bold=True,color="#B9450A")
line("sample_label_leader",[594,550,568,518],col="#BD4B0C",width=1,arrow=True)
text("receiver_label","Ground receiver",[179,703,190,38],fs=18.5)
line("receiver_label_leader",[322,741,353,761],col=navy,width=1,arrow=True)
# Panel b
text("panel_b","(b)",[1065,5,65,43],r=1,fs=25.5,bold=True)
text("interpolation_heading","Local bilinear interpolation",[1125,38,493,42],r=1,fs=23.25,bold=True)
shape("era5_grid","rect",[1179,151,344,219],r=1,fill="#E5F2F4",stroke=navy,width=1.1,z=10)
line("grid_lat_construction",[1179,250,1523,250],r=1,col="#79AFC4",width=.8,dash="dash",z=14)
line("grid_lon_construction",[1395,151,1395,370],r=1,col="#79AFC4",width=.8,dash="dash",z=14)
for name,x,y,tx,ty in [("1",1179,151,1157,110),("2",1523,151,1529,110),("3",1179,370,1152,378),("4",1523,370,1526,378)]:
    shape("G"+name,"ellipse",[x-8,y-8,16,16],r=1,fill="#3899EB",stroke=navy,width=1,z=28)
    text("G"+name+"_label","G"+chr(8320+int(name)),[tx,ty,44,35],r=1,fs=21)
shape("interpolation_target","ellipse",[1387,238,22,22],r=1,fill=orange,stroke=navy,width=1,z=30)
# Leave clear patch only where original label obscured the helper line.
shape("intersection_label_white","rect",[1444,218,178,40],r=1,fill="#FFFFFF",stroke="none",z=35)
text("intersection_label","Path intersection",[1447,218,180,35],r=1,fs=18,color="#BE4B12")
line("intersection_leader",[1445,241,1415,249],r=1,col="#BE4B12",width=1,arrow=True,z=36)
line("latitude_axis",[1135,393,1135,145],r=1,col=navy,width=1.5,arrow=True)
line("longitude_axis",[1192,413,1535,413],r=1,col=navy,width=1.5,arrow=True)
text("latitude_label","Latitude",[1063,243,97,33],r=1,fs=19,rotation=-90,align="center")
text("longitude_label","Longitude",[1269,413,170,37],r=1,fs=19,align="center")
text("grid_caption","Four neighboring ERA5 grid points",[1194,458,415,38],r=1,fs=18.3)
# Panel c
text("panel_c","(c)",[1038,525,65,43],r=2,fs=25.5,bold=True)
text("inputs_heading","Retained meteorological inputs",[1125,545,500,44],r=2,fs=23.25,bold=True)
shape("base_card","roundRect",[1031,604,304,159],r=2,fill="#E8F5E5",stroke="#4DAB54",width=1,z=10,source_corner_radius_px=14,corner_category="small-radius")
shape("profile_card","roundRect",[1352,610,269,151],r=2,fill="#E1F0FB",stroke="#3896EB",width=1,z=10,source_corner_radius_px=14,corner_category="small-radius")
text("base_title","50 base features",[1041,616,284,40],r=2,fs=21.5,bold=True,color="#133D1C",align="center")
text("base_surface","Surface quantities",[1042,656,282,34],r=2,fs=18.5,align="center")
text("base_diagnostic","Single-level diagnostics",[1039,686,288,32],r=2,fs=18.5,align="center")
text("base_column","Column-integrated quantities",[1037,714,292,32],r=2,fs=17.6,align="center")
text("profile_title","106 pressure-level\nfeatures",[1361,622,251,60],r=2,fs=21,bold=True,align="center")
text("profile_body","Variable-level pairs across\n10 levels",[1362,689,249,56],r=2,fs=17.7,color="#245C87",align="center")
# Smooth editable brace joining both cards.
bez("join_brace",[[1135,768,1135,780,1136,784,1150,784],
                  [1150,784,1220,784,1290,784,1320,784],
                  [1320,784,1330,784,1332,791,1332,800],
                  [1332,800,1332,790,1337,784,1345,784],
                  [1345,784,1390,784,1485,784,1520,784],
                  [1520,784,1532,784,1534,778,1534,767]],r=2,col="#6F859A",width=1.5,z=20)
shape("total_card","roundRect",[1158,807,346,54],r=2,fill="#FFFFFF",stroke="#101C3B",width=1.3,z=10,source_corner_radius_px=10,corner_category="small-radius")
text("total_count","50 + 106 = 156 features",[1165,812,332,43],r=2,fs=21.4,bold=True,align="center")
text("base_only_label","Base-only candidate pool: 50",[1187,868,400,38],r=2,fs=18.6,bold=True,color="#17471A")
text("scale_note","Local sampling schematic; geometry not to scale.",[1220,923,402,32],r=2,fs=12.4,align="right")
# Inventories and measurement anchors for all native scientific objects.
for r in regions:
    m["visual_inventory"].append({"id":"native_"+r["id"],"kind":"native-structure","representation":"native","description":r["label"]+": editable text, geometry and associated labels"})
# Compound geometry: measured point nodes and straight directed/undirected relationships.
for ix in [0,1,2]:
    r=regions[ix]; ids=set(r["manifest_ids"]["shapes"])
    nodes=[];edges=[]
    for s in m["shapes"]:
        if s["id"] not in ids:continue
        if s["type"]=="ellipse" or (ix==2 and s["type"]=="roundRect"):
            x,y,w,h=s["box_px"];nodes.append({"id":s["id"],"manifest_id":s["id"],"geometry":"circle" if s["type"]=="ellipse" else "rect",
            "source_center_px":[x+w/2,y+h/2],"source_size_px":[w,h],"position_tolerance_px":1,"size_tolerance_px":1})
        if s["type"]=="line":
            edges.append({"id":s["id"],"manifest_id":s["id"],"source_points_px":s["points_px"],
            "line_style":"dashed" if s.get("dash") else "solid","direction":"end" if s.get("end_arrow") else "none"})
    r["compound_diagram"]={"kind":"compound-diagram","object_inventory_complete":True,"measurement_reviewed":True,
      "native_object_policy":"measured-simple-objects-native","nodes":nodes,"edges":edges}
m["image2ppt_region_decomposition"]={"schema_version":"image2ppt-region-decomposition-v1","page_complexity":"structured",
 "source_size_px":[1634,962],"regions":regions}
(P/"manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"text_boxes":len(m["text_boxes"]),"native_shapes":len(m["shapes"]),"picture_assets":len(m["images"])}))

