# remapy
Playing with tools inspired by Remy's therapies

**NOTE: This is a work in progress**

Remy + therapy + python = remapy

Much of this is implemented using Claude code and other coding LLMs. I build similar tools to sample sensors from embedded systems and process them in my day job. 

## Motivation

My son Remy has an ultra-rare genetic syndrome that causes global developmental
delays, in both movement and cognitive function (more at
[rareremy.org](https://www.rareremy.org)). While we pursue therapeutic development
and medical research, most of our day-to-day energy goes into physical and
occupational therapy with him.

remapy is a place to build tools that motivate Remy in those therapies and that can
track changes in his abilities before they're visible in his gross actions. If
therapeutic development advances to a clinical trial, the same tools can help
establish a baseline of his abilities before treatment and measure progress or
change afterward.

![Pose estimation on a session with Remy](img/remy_pose.png)

## References

Standard, clinically validated exercises and scoring systems this project draws on
for motor-development metrics:

- **GMFM-88 — Gross Motor Function Measure.** 88 items across five dimensions
  (lying/rolling, sitting, crawling/kneeling, standing, walking/running/jumping),
  each scored 0–3.
  [CanChild overview](https://canchild.ca/en/resources/44-gross-motor-function-measure-gmfm) ·
  [Physiopedia](https://www.physio-pedia.com/Gross_Motor_Function_Measure) ·
  [User's Manual (Mac Keith Press)](https://www.mackeith.co.uk/book/gross-motor-function-measure-gmfm-66-gmfm-88-users-manual-revised-3rd-edition/)
- **PDMS-3 — Peabody Developmental Motor Scales, Third Edition.** Gross-motor
  subtests (Body Control, Body Transport, Object Control) plus fine-motor and a
  supplemental physical-fitness subtest.
  [Pearson](https://www.pearsonassessments.com/en-us/Store/Professional-Assessments/Motor-Sensory/Peabody-Developmental-Motor-Scales,-Third-Edition/p/P100049000) ·
  [WPS](https://www.wpspublish.com/peabody-developmental-motor-scales-third-edition.html) ·
  [PAR](https://www.parinc.com/products/PDMS-3)
- **AIMS — Alberta Infant Motor Scale.** 58 observational items across prone,
  supine, sitting, and standing positions, norm-referenced from birth to 18 months.
  [Physiopedia](https://www.physio-pedia.com/Alberta_Infant_Motor_Scale_(AIMS)) ·
  [Score sheets (Elsevier)](https://www.us.elsevierhealth.com/alberta-infant-motor-scale-score-sheets-aims-9780323798426.html)

## TODO

- [ ] implement metrics from standard exercises and scoring defined in GMFM-88, Peabody Developmental Motor Scales (PDMS-3), and Alberta Infant Motor Scale (AIMS): real-time and offline with notebook exploration
- [ ] measure streaming throughput compared to nominal sampling rates
- [ ] baseline IMU signal stats at rest
- [ ] set LED color to show battery level high, mid, or low.
- [ ] use embedded system with real hardware interrupts - rpi pico 2w with external sensors.
- [ ] port to Android 
- [ ] consider multiple camera views
- [ ] consider adding a depth camera
