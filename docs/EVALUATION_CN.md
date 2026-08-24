# 评价流程

## 三层证据

1. **抓取随动**：跟踪闭合附近布片相对TCP的行程、方向和分离增长。
2. **折叠语义**：检查固定主体位移、折片覆盖率和层叠关系。
3. **人工四视角**：overview、overhead、shirt_bottom、right_grasp同步复核。

只看最终俯视图不能证明中途没有夹空、穿桌、U形兜或错误布层。

## 关键帧

- 第一折：45、54、90、117、150、180、210、300、332
- 第二折：385、393、430、439、480、505、530、550、570、583
- 第三折：650、680、690、720、750、780、840、900、920、941、960、979

## 自动诊断

```bash
GENESIS_PYTHON=/path/to/python \
  ./diagnostics/run_diagnostics.sh outputs/episode_000000/<run>.json
```

输出包括provenance、语义状态、manual review模板及evaluation JSON/Markdown。
自动Gate是护栏，不替代人工视频验收。

## 当前主要Gate

- 每次抓取：布片与TCP同向共动，不能靠接近距离或最终形状假判成功；
- 第二折：中央主体不应整体拖走，左右折片与base应有合理覆盖；
- 第三折：固定上片中位位移小于30 mm，bottom/top覆盖率大于45%；
- 最终：全部释放、无明显穿透、无永久U兜，折形在沉降后保持。

第三折当前虽然最终coverage PASS，但抓取分离增长约90 mm，因此状态是
“功能完成、抓取质量FAIL”，不能简写成五抓全PASS。

## 公平比较规则

- 对dHat、材料或网格下结论时，尽量保持其他配置一致；
- 区分严格SIM1同轨迹后端比较与成功导向Genesis Demo；
- 记录代码commit、Genesis patch hash、SIM1 commit、资产hash、GPU和solver；
- 最终候选至少重复3次后再宣称鲁棒。
