# 实现修订 v002 — NCAR TCLW 名称差异

2026-10-16，/root。v001 在首月TCLW DAS的long_name检查处失败，尚未开始60个气象子集下载。失败收据和2673字节原始part保留，SHA为1113fe0cdebab0547edba57146b21147f9ebe94a3329ce3a5882e4c8d5bf34f6。此前4份DAS成功收据复用，不重取。

实读官方 NCAR DAS：TCLW / short_name=tclw / ECMWF local table128 parameter78 / kg m**-2 / long_name="Total column liquid water"。该历史名称与[ECMWF参数78](https://codes.ecmwf.int/grib/param-db/78)当前的Total column cloud liquid water、[ERA5官方参数表](https://confluence.ecmwf.int/pages/viewpage.action?pageId=179741903)的tclw=78对应。它不是TCW参数136、TCWV参数137或总云量164；参数78指云滴液态水，不包含降雨水滴。

修复仅在新wrapper中把TCLW预期源long_name改为实际的精确字符串。参数表/号、short_name、单位、坐标、UTC、形状、有限值、网格与所有其它门槛不变；标准化输出列仍为surface__total_column_cloud_liquid_water。独立验证wrapper同步其独立原始名称合同，不改变数值比较。

旧acquire_physical_fields.py、verify_physical_fields.py、scope_plan.json、sources_before.json、失败记录均不修改。新采用入口acquire_physical_fields_v002.py，仅支持record-fix、acquire，不能prepare。验证入口verify_physical_fields_v002.py。implementation_v002.json保存本次新增实现SHA和解释；继续前后双重核验原保护源及新文件。原12时间块/60子集/5664小时/28320值不变；仍无训练、模型推理或旧下载器恢复。

此前名称失败没有被视为成功。原part原样保留，修复依据明确后同一DAS URL重新取得并全验证，才晋级正式元数据。
