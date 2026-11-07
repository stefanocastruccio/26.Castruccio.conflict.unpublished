# Literature check: bivariate conflict-unrest model (2026-11-07)

DOIs verified against Crossref by the search agent; arXiv items unverified.

I found no published paper that already does what you plan. Nothing I located jointly models protest/riot and conflict occurrence with a shared space-time Matérn field and cross-lagged terms. Every DOI below was checked against Crossref. The arXiv items have no DOI, and I couldn't open their pages because arxiv.org isn't on this session's allowed network list, so treat what I say about them as unconfirmed.

## (1) Multivariate or joint spatio-temporal models of conflict events
- **Browning, R., Patten, ?, Rousseau, J., Mengersen, K. (2026).** Bayesian spatiotemporal modelling of political violence and conflict events using discrete-time Hawkes processes. *JRSS A*. doi:10.1093/jrsssa/qnag039. Fits a discrete-time Hawkes model with a spatially varying baseline to ACLED counts, including protests, in South Asia. It is the closest statistics competitor, but event types are fitted one at a time and the dependence is self-excitation, not a shared latent field.
- **Jun, M., Cook, S. J. (2024).** Flexible multivariate spatiotemporal Hawkes process models of terrorism. *Ann. Appl. Stat.* 18(2), 1378–1403. doi:10.1214/23-AOAS1839. A bivariate Hawkes model with nonseparable cross-triggering between groups (Nigeria). It is the AoAS precedent for a "cross-excitation" reading. It explains co-occurrence only through triggering and has no shared latent component, so you should contrast your design with it directly.
- **Racek, D., Thurner, P. W., Kauermann, G. (2025).** Capturing the spatio-temporal diffusion effects of armed conflict: a nonparametric smoothing approach. *JRSS A* 189, 1428–1447. doi:10.1093/jrsssa/qnaf120. Puts Hawkes-like diffusion terms into the predictor of a GAM on gridded conflict data. It is a close analogue of your neighbour-lag terms, but for a single outcome.
- **Zammit-Mangion, A., Dewar, M., Kadirkamanathan, V., Sanguinetti, G. (2012).** Point process modelling of the Afghan War Diary. *PNAS* 109, 12414–12419. doi:10.1073/pnas.1203177109. A spatio-temporal LGCP for conflict intensity, with one outcome.
- **Python, A., Illian, J. B., Jones-Todd, C. M., Blangiardo, M. (2019).** A Bayesian approach to modelling subnational spatial dynamics of worldwide non-state terrorism. *JRSS A* 182, 323–344. doi:10.1111/rssa.12384. Bernoulli and Poisson SPDE/INLA space-time models. Same toolchain as yours, but univariate.
- **Methods for the shared field:** Gelfand et al. (2004), *Test* 13, 263–312, doi:10.1007/bf02595775, for the LMC. Palmí-Perales, Gómez-Rubio, Martinez-Beneito (2021), *JSS* 98, doi:10.18637/jss.v098.i02, for multivariate INLA models. Diggle et al. (2013), *Stat. Sci.* 28, 542–563, doi:10.1214/13-sts441, for LGCPs. I found no conflict application of an LMC or "copy" field.

## (2) Escalation from protest to violence
- **Rød, E. G., Weidmann, N. B. (2023).** From bad to worse? How protest can foster armed conflict in autocracies. *Political Geography* 103, 102891. doi:10.1016/j.polgeo.2023.102891. A local spatial design finding that protest raises the risk of armed conflict mainly through repression and "protest capture". This is the substantive claim you are testing, but it uses regression with no latent confounding field.
- **Rød, E. G., Hegre, H., Leis, M. (2025).** Predicting armed conflict using protest data. *J. Peace Res.* 62(1), 3–20. doi:10.1177/00223433231186452. Uses the same 0.5° PRIO-GRID cells, protest counts and neighbour spatial lags. Your cross-lagged terms resemble its design; the difference is that its aim is forecasting, not decomposition.
- **Ryckman, K. C. (2020).** A turn to violence: the escalation of nonviolent movements. *JCR* 64, 318–343. doi:10.1177/0022002719861707. Campaign-level and not spatial; it is the theory reference for nonviolent-to-violent transitions.
- **Ritter, E. H., Conrad, C. R. (2016).** Preventing and responding to dissent. *APSR* 110, 85–99. doi:10.1017/S0003055415000623. Argues that dissent and repression are endogenous to each other, and uses a rainfall instrument with SCAD. This is the argument for why a shared latent component is needed.
- Also usable: Chenoweth, Perkoski & Kang (2017), *JCR* 61, 1950–1969, doi:10.1177/0022002717721390; Sullivan (2016), *JCR* 60, 1163–1190, doi:10.1177/0022002714567951; Carey (2006), *PRQ* 59, 1–11, doi:10.1177/106591290605900101. None of these uses a spatial statistical model.

## (3) Reporting bias
- **Weidmann, N. B. (2016).** A closer look at reporting bias in conflict event data. *AJPS* 60, 206–218. doi:10.1111/ajps.12196. Shows that reporting rates vary in space, for example with cellphone coverage. This is the main reason to give the shared field a "reporting intensity" reading.
- **Weidmann, N. B. (2015).** On the accuracy of media-based conflict event data. *JCR* 59, 1129–1149. doi:10.1177/0022002714530431.
- **Cook, S. J., Blas, B., Carroll, R. J., Sinha, S. (2017).** Two wrongs make a right: addressing underreporting in binary data from multiple sources. *Political Analysis* 25, 223–240. doi:10.1017/pan.2016.13. Separates true occurrence from reporting using several *sources* (SCAD), not several event types. It is the closest formal model of reporting, and it identifies reporting from multiple sources, which your design does not have.
- **Cook, S. J., Weidmann, N. B. (2019).** Lost in aggregation. *AJPS* 63, 250–264. doi:10.1111/ajps.12398. Also Cook & Weidmann (2022), Race to the bottom: spatial aggregation and event data, *Int. Interactions* 48, 471–491, doi:10.1080/03050629.2022.2025365. Relevant to your choice of a 0.5° weekly grid.
- **Hellmeier, S., Weidmann, N. B., Rød, E. G. (2018).** In the spotlight: analyzing sequential attention effects in protest reporting. *Political Communication* 35, 587–611. doi:10.1080/10584609.2018.1452811. Finds that reporting itself clusters in space and time. This is a direct rival explanation to "escalation" in your cross-lags.
- **Croicu, M., Eck, K. (2022).** Reporting of non-fatal conflict events. *Int. Interactions* 48, 450–470. doi:10.1080/03050629.2022.2044325. Non-fatal events, which includes most protests, are under-reported in a selective way.

## (4) Recent statistical analyses of ACLED (2020–2026)
- **Egbon, O. A., et al. (2024).** Bayesian spatio-temporal statistical modeling of violent-related fatality in western and central Africa. *Spatial Statistics* 60, 100828. doi:10.1016/j.spasta.2024.100828. A zero-inflated SPDE model with one outcome type.
- **Egbon, O. A., Gayawan, E. (2025).** Spatio-temporal modeling of violent conflict and fatality in Nigeria: a point process modeling with SPDE approach. *Stat. Politics Policy* 16, 63–86. doi:10.1515/spp-2024-0028.
- **Kushwaha, N., Lee, E. D. (2023).** Discovering the mesoscale for chains of conflict. *PNAS Nexus* 2, pgad228. doi:10.1093/pnasnexus/pgad228. Grid-cell spread of ACLED battles.
- **Preprints, not verified:** arXiv 2511.00867 and 2511.12219 (Ethiopia, INLA). arXiv 2603.25964, "Assessing Reporting Delays in ACLED", is relevant to the reporting-intensity reading.
- I found nothing on ACLED in Environmetrics or JASA, and no ACLED protest–conflict joint model in AoAS.

## Assessment
**Already done:**
- Univariate SPDE/INLA models of conflict data (Python et al.; Egbon et al.).
- Multivariate cross-excitation between conflict streams (Jun & Cook, AoAS).
- Hawkes models of ACLED protests (Browning et al.).
- Protest-to-conflict effects and forecasting with neighbour lags on a 0.5° grid (Rød & Weidmann; Rød, Hegre & Leis).
- Correcting reporting with multiple sources (Cook et al.).
- LMC models in INLA, as general methodology.

**Genuinely new, as far as I can find:**
1. A joint latent Gaussian model of protest/riot and conflict occurrence with a shared space-time AR(1)×Matérn field and a loading. I found no conflict-data use of an LMC or "copy" field.
2. Using cross-lags and a shared field *together* to separate escalation from co-occurring latent episodes. The Hawkes papers and the Rød papers each fold the two together.
3. Reading the shared field as reporting intensity, using two event types from one source rather than several sources.

The third point is the weakest. With one source you cannot separate reporting intensity from a real joint shock without extra assumptions. Reviewers will cite Cook et al. (2017) and Hellmeier et al. (2018) on this, so present it as a partial or sensitivity-based interpretation. You should also show that you can recover the cross-lags when a shared field is present, for example by simulation.

**Must cite (10):**
- Jun & Cook 2024
- Browning et al. 2026
- Rød & Weidmann 2023
- Rød, Hegre & Leis 2025
- Weidmann 2016
- Cook, Blas, Carroll & Sinha 2017
- Hellmeier, Weidmann & Rød 2018
- Ritter & Conrad 2016
- Racek, Thurner & Kauermann 2025
- Python et al. 2019, or Egbon et al. 2024 for SPDE models of ACLED

Add Gelfand et al. 2004 for the LMC.

Sources:
- [Browning et al., JRSS A](https://doi.org/10.1093/jrsssa/qnag039)
- [Jun & Cook, AoAS (Project Euclid)](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-18/issue-2/Flexible-multivariate-spatiotemporal-Hawkes-process-models-of-terrorism/10.1214/23-AOAS1839.full)
- [Racek et al., JRSS A](https://academic.oup.com/jrsssa/article/189/3/1428/8216735)
- [Rød, Hegre & Leis, JPR](https://journals.sagepub.com/doi/10.1177/00223433231186452)
- [Rød & Weidmann (KOPS)](https://kops.uni-konstanz.de/entities/publication/60017d5c-a261-40fe-88d8-da8e3de8a967/full)
- [Weidmann, AJPS](https://ajps.org/2015/07/27/a-closer-look-at-reporting-bias-in-conflict-event-data/)
- [Cook et al., Political Analysis](https://cambridge.org/core/journals/political-analysis/article/two-wrongs-make-a-right-addressing-underreporting-in-binary-data-from-multiple-sources/CA369D0E7446D591EB88E1D37521FA36)
- [Hellmeier et al. (KOPS)](https://kops.uni-konstanz.de/handle/123456789/42971)
- [Python et al. (St Andrews)](https://research-portal.st-andrews.ac.uk/en/publications/a-bayesian-approach-to-modelling-subnational-spatial-dynamics-of-/)
- [Ritter & Conrad (IDEAS)](https://ideas.repec.org/a/cup/apsrev/v110y2016i01p85-99_00.html)
- [Zammit-Mangion et al. (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3411944)
- [Egbon & Gayawan, De Gruyter](https://www.degruyterbrill.com/document/doi/10.1515/spp-2024-0028/html)
- [Kushwaha & Lee coverage (CSH)](https://csh.ac.at/news/scientists-create-method-to-spot-the-spread-of-armed-conflicts/)
- [ACLED reporting delays (arXiv)](https://arxiv.org/pdf/2603.25964)
- [Ethiopia INLA (arXiv)](https://arxiv.org/html/2511.00867)
- [Crossref API](https://api.crossref.org)