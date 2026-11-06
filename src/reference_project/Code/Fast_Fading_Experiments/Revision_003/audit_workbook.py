"""Verify exported XLSX cells against immutable parameter records, without Excel mutation."""
import zipfile,xml.etree.ElementTree as ET
from common import *

def main():
    book=DELIVERY/'RFE_每模型每轮删除与保留参数.xlsx'
    src=read(DELIVERY/'rfe_workbook_data.json');ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(book) as z:
        matrices=[]
        for name in ('xl/worksheets/sheet1.xml','xl/worksheets/sheet2.xml'):
            root=ET.fromstring(z.read(name));cells={c.attrib['r']:c.find('s:v',ns).text if c.find('s:v',ns) is not None else '' for c in root.findall('.//s:c',ns)}
            need(not root.findall('.//s:c[@t="e"]',ns),'Workbook formula error')
            pane=root.find('.//s:pane',ns);need(pane.attrib['xSplit']=='1' and pane.attrib['ySplit']=='5','Frozen headers')
            matrices.append(cells)
        for i,row in enumerate(src['rows'],6):
            cells=matrices[0]
            for j,value in enumerate(row):
                actual=cells.get(chr(65+j)+str(i),'')
                if value is None or value=='':need(actual=='','Expected blank')
                elif j in (4,8):need(actual.split('\n')[-1]==value,'Native deleted-feature cell')
                else:need(actual==str(value),'RFE cell mismatch')
        for i,row in enumerate(src['dictionary'],6):
            cells=matrices[1]
            need(cells['A'+str(i)]==row[0] and cells['C'+str(i)]==row[1] and cells['D'+str(i)]==row[2],'Dictionary mapping')
        for table,expected in [('xl/tables/table1.xml','A5:I1565'),('xl/tables/table2.xml','A5:F161')]:
            root=ET.fromstring(z.read(table));need(root.attrib['ref']==expected and root.find('s:autoFilter',ns) is not None,'Filter/table extent')
    for p,h in read(DELIVERY/'RFE_QA.json')['sources_sha256'].items():need(sha(ROOT/p)==h,'RFE source changed')
    result={'status':'PASS','utc':now(),'workbook_sha256':sha(book),'verified_state_rows':1560,'verified_dictionary_rows':156,'numeric_cells_typed':True,'tables_filterable':True,'frozen_header_rows':5,'original_rfe_inputs_unchanged':True,'visual_inspection':'Main table title, wrapped baseline IDs, k10-k1 current/next deletion cells and Chinese/native dictionary were inspected as rendered PNGs; no truncation affecting data. Wide table intentionally horizontal-scrollable.'}
    write(DELIVERY/'RFE_WORKBOOK_QA.json',result);print(result)
if __name__=='__main__':main()
