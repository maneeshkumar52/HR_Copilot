"""
create_sample_data.py
Creates all sample HR policy documents and structured data files.
Run directly or called from setup.sh / setup.bat
"""
import os, json, csv

print("Creating sample HR knowledge base...")

os.makedirs("data/hr_docs", exist_ok=True)
os.makedirs("data/hr_structured", exist_ok=True)
os.makedirs("data/index", exist_ok=True)
os.makedirs("data/eval", exist_ok=True)

# ── 6 HR Policy Documents ─────────────────────────────────────────────────────
# NOTE: Documents are intentionally detailed (~6000-8000 chars each) so the
# HR-aware chunker produces 5-8 meaningful chunks per document (vs 1 chunk
# per document with very short files). This is critical for RAG quality.

DOCS = {

"data/hr_docs/leave_policy.md": """# Leave Policy — Enterprise Corp (Effective Jan 2026)

## Overview
Enterprise Corp's Leave Policy ensures every employee can take time off for rest, personal commitments, health, and family events while maintaining business continuity. This policy is effective from January 1, 2026 and supersedes all previous leave guidelines.

## Annual Leave (Privilege Leave)
All permanent employees are entitled to **24 working days** of annual leave per calendar year.

### Accrual
- Leave accrues at the rate of **2 days per month** from the date of joining.
- New joiners receive prorated leave in their first year based on joining month.
- Leave balance is updated on the 1st of each month.

### Carry-Forward Rules
- **Maximum carry-forward cap: 8 days.** Any balance above 8 days lapses on **31 March** each year.
- Employees are strongly advised to plan and utilize leave within the year.
- HRBP approval required to carry forward more than 8 days in exceptional circumstances (medical emergency, project criticality).

### Leave Encashment
- Up to **15 days** of accumulated annual leave can be encashed once per year.
- Encashment window: **1-31 December** each year.
- Encashment is calculated at Basic Salary / 26 working days per day.
- Employees who resign or are terminated will receive encashment for remaining annual leave as part of Full & Final settlement.
- Encashment is NOT allowed while the employee is on notice period — accumulated leave is adjusted against notice period instead.

### Minimum Leave Balance
- Employees must maintain a minimum of 3 days leave balance at all times.
- Negative leave is not permitted. Employees must ensure sufficient balance before applying.

## Casual Leave
- **6 days per calendar year** for all permanent employees.
- Casual leave is for short, unplanned absences — personal errands, minor illness, urgent personal matters.
- **Non-carry-forward**: Unused casual leave lapses on 31 December each year. No encashment.
- Casual leave **cannot be combined** with annual leave in a single application. They must be applied separately with a gap.
- Maximum 3 consecutive days can be taken as casual leave without escalation.

## Sick Leave
- **12 days per calendar year** for all permanent employees.
- Sick leave is for personal illness, medical appointments, or health-related absences.
- A medical certificate from a registered doctor is **mandatory for more than 3 consecutive sick days**.
- Extended medical leave beyond 12 days requires written approval from the HR Business Partner (HRBP) and a treating doctor's report.
- Sick leave does not carry forward and cannot be encashed. Unused sick leave lapses on 31 December.
- Employees may be asked to attend a Return to Work interview after any sick leave of 7+ consecutive days.
- Sick leave cannot be combined with annual leave in the same leave application period.

## Maternity Leave
- **26 weeks (182 calendar days)** fully paid maternity leave as per the Maternity Benefit Act, 1961 (amended 2017).
- Applicable to female employees who have worked for at least **80 days** in the preceding 12 months.
- For a third child and beyond, maternity leave is 12 weeks.
- Adoption leave: **12 weeks** for the primary caregiver (adoption of a child below 3 months of age).
- Commissioning mothers (surrogacy): **12 weeks** leave from the date the child is handed over.
- Maternity leave can start up to 8 weeks before the expected date of delivery.
- Work from home arrangement may be offered after maternity leave for up to 3 months on mutual agreement.

## Paternity Leave
- **10 working days** fully paid paternity leave.
- Must be taken **within 3 months** of the child's birth or legal adoption.
- Can be split into two blocks (minimum 3 days per block).
- Applicable for the first two children only.
- Paternity leave can be combined with annual leave for a longer continuous break.

## Bereavement Leave
- **Immediate family** (spouse, parent, child, sibling): **5 working days**
- **Extended family** (grandparent, parent-in-law, sibling-in-law): **3 working days**
- Bereavement leave should be applied in HRMS and supporting documentation (death certificate or equivalent) submitted within 7 working days.
- Additional leave beyond the entitlement above can be taken as annual leave or leave without pay with manager approval.

## Marriage Leave
- **5 working days** for the employee's own marriage.
- Must be applied at least 2 weeks in advance.
- Can be combined with annual leave for a longer honeymoon break.
- Applicable once during employment at Enterprise Corp.

## Leave Without Pay (LWP)
- Employees who have exhausted all leave balances may apply for Leave Without Pay.
- LWP deduction is calculated at Basic Salary / 26 per day and applied in the current month's payroll.
- Approval from HRBP required. LWP impacts variable pay calculation for that quarter.

## Compensatory Off (Comp-Off)
- Employees who work on a declared public holiday or a weekend due to business requirements are eligible for compensatory off.
- Comp-off must be approved by the manager before the extra workday is undertaken.
- Comp-off must be utilized within **60 days** of being earned; it lapses thereafter.
- Comp-off cannot be encashed.

## Leave Application Process
1. **Apply on the HRMS portal** minimum **48 hours in advance** for all planned leave.
2. **Emergency leave**: Notify your reporting manager via phone or WhatsApp within **2 hours** of starting the absence. Apply on HRMS retroactively by the next working day.
3. **Manager approval**: Required for all leaves exceeding **3 consecutive days**.
4. **Team coverage plan**: Mandatory for leaves of **5+ consecutive days** — the employee must arrange coverage and document it in the leave application notes.
5. **Leave cancellation**: Can be done on the HRMS portal up to 24 hours before the leave start date without penalty.
6. HR may reject leave applications in consultation with the manager if critical project deadlines are at risk.

## Combining Leave Types
- Annual leave and sick leave **cannot** be combined in a single continuous leave period.
- Casual leave and annual leave **cannot** be combined in a single leave application.
- Annual leave **can** be combined with public holidays to create extended long weekends.
- Paternity leave **can** be combined with annual leave to create a longer continuous break.
- Marriage leave **can** be combined with annual leave for an extended holiday.

## Public Holidays
- **13 declared national/festival public holidays** per year published by HR in January each year.
- **3 optional restricted holidays** can be selected by each employee from a list of 10 regional or religious holidays.
- Optional restricted holidays must be applied via HRMS at least 48 hours in advance.
- Working on a public holiday requires manager approval in advance and entitles the employee to a compensatory off.

## Leave Policy for Probationers
- Probationers (first 6 months of employment) are entitled to **6 days sick leave** and **3 days casual leave** only during probation.
- Annual leave begins accruing from the date of joining but cannot be encashed during probation.
- Probationers are not eligible for marriage leave, bereavement leave enhancements, or comp-off during probation.

## Contact for Leave Queries
- HR Business Partner: hrbp@enterprise.com
- HR Helpline: ithelpdesk@enterprise.com (ext. 1234)
""",

"data/hr_docs/compensation_benefits.md": """# Compensation & Benefits Guide — Enterprise Corp 2026

## Overview
This guide covers the complete compensation structure, benefits, allowances, and incentives available to Enterprise Corp employees. All figures are effective from April 1, 2026 unless otherwise stated.

## Salary Structure
Enterprise Corp follows a Cost to Company (CTC) model with a fixed and variable split.

- **Fixed component: 70% of CTC** — paid monthly, includes basic salary and allowances.
- **Variable component: 30% of CTC** — paid quarterly based on OKR score and company performance.
- Variable pay release timeline: April, July, October, January (for the preceding quarter).
- If OKR score is below 50%, variable pay for that quarter is not released.

## Salary Bands
Enterprise Corp uses 6 salary bands covering all roles from Associate to VP level.

| Band | Title | CTC Range | Notice Period | ESOP |
|------|-------|-----------|---------------|------|
| B1 | Associate | Rs 4,00,000 - Rs 7,00,000 | 1 month | Not eligible |
| B2 | Senior Associate | Rs 7,00,000 - Rs 12,00,000 | 2 months | Not eligible |
| B3 | Lead / Senior Lead | Rs 12,00,000 - Rs 20,00,000 | 2 months | Not eligible |
| B4 | Manager / Senior Manager | Rs 20,00,000 - Rs 35,00,000 | 3 months | Eligible |
| B5 | Director | Rs 35,00,000 - Rs 60,00,000 | 3 months | Eligible |
| B6 | VP / Senior Director | Rs 60,00,000 - Rs 1,20,00,000 | 3 months | Eligible |

Band 3 Senior Lead CTC range is Rs 12,00,000 to Rs 20,00,000 per annum with a 2-month notice period.
Band 4 Manager salary range is Rs 20,00,000 to Rs 35,00,000 per annum with ESOP eligibility and 3-month notice.

## Annual Increment Cycle
- **Increment cycle: April annually.** Effective from April 1 each year.
- Increment range: **0% to 25%** based on the annual performance rating (scale of 1 to 5).
- Employees who join after October 1 receive a prorated increment in their first cycle.
- Employees on PIP (Performance Improvement Plan) during the appraisal cycle are not eligible for increment.

## Performance Ratings and Increment Matrix

| Rating | Label | Annual Increment Range |
|--------|-------|------------------------|
| 5 | Exceptional | 18% - 25% |
| 4 | Exceeds Expectations | 12% - 18% |
| 3 | Meets Expectations | 6% - 12% |
| 2 | Needs Improvement | 0% - 5% |
| 1 | Does Not Meet Expectations | 0% (auto-PIP initiated) |

Performance ratings are finalized by the manager and calibrated by the HRBP in March each year.

## Monthly Benefits — Tax-Exempt Components
These components are part of the fixed CTC and are tax-exempt under Income Tax rules.

- **Meal Allowance**: Rs 2,200 per month (exempt under Section 10 of Income Tax Act)
- **Broadband / Internet Reimbursement**: Rs 1,500 per month — reimbursed against actual bills
- **Children Education Allowance**: Rs 200 per month per child (maximum 2 children)
- **Transport Allowance**: Rs 1,600 per month (metro cities) or Rs 800 per month (other locations)
- **Medical Allowance**: Rs 1,250 per month (reimbursable against medical bills)

## Annual Benefits
These benefits are provided annually and are over and above the CTC for most components.

- **Learning & Development Budget**: Rs 25,000 per employee per year (resets April 1)
- **Health Insurance (Mediclaim)**: Rs 5,00,000 floater policy covering self, spouse, and up to 2 dependent children. Cashless hospitalization at 5,000+ network hospitals.
- **Term Life Insurance**: Coverage of 3x annual CTC. AD&D cover included.
- **Personal Accident Insurance**: Rs 25,00,000 coverage for accidents.
- **Gratuity**: Payable after minimum 5 years of continuous service, as per the Payment of Gratuity Act.
- **ESOP (Employee Stock Option Plan)**: Band 4 and above are eligible. 4-year vesting schedule with a 1-year cliff.

## ESOP Details — Band 4 and Above
- ESOP grants are made at joining and during annual appraisal cycles.
- Vesting schedule: 25% after 1 year (cliff), then monthly over the remaining 3 years.
- Options are exercisable within 5 years of vesting.
- ESOP eligibility begins at Band 4 (Manager / Senior Manager) and all higher bands.
- Bands B1, B2, and B3 are NOT eligible for ESOP.

## Bonus and Incentives
- **Annual Performance Bonus**: 10% to 20% of annual fixed pay. Paid in April along with increment.
- **Spot Award**: Up to Rs 20,000 for exceptional one-time contributions.
- **Referral Bonus**: Rs 30,000 for successful lateral hire referrals. Paid after the referred employee completes 6 months.
- **Sales Incentive Plan (SIP)**: Additional commission structure for sales roles, separate from standard variable pay.

## Provident Fund (PF)
- **Employee contribution**: 12% of Basic Salary per month (deducted from salary).
- **Employer contribution**: 12% of Basic Salary per month (added by company).
- PF is deposited with EPFO by the 15th of each month.
- PF can be withdrawn after 5 years of continuous service tax-free.

## Payroll and Pay Slips
- Salary is credited to registered bank accounts on the **last working day** of each month.
- Pay slips are available on the HRMS portal by the 3rd of the following month.
- Tax deductions (TDS) are calculated based on the tax declaration submitted at the start of the financial year.
- Employees must submit tax-saving investment proofs by January 31 each year.

## Contact for Compensation Queries
- Payroll Team: payroll@enterprise.com
- HRBP: hrbp@enterprise.com
""",

"data/hr_docs/remote_work_policy.md": """# Remote Work & Flexible Working Policy — Enterprise Corp 2026

## Overview
Enterprise Corp supports flexible and hybrid working arrangements to improve employee well-being and productivity while maintaining team collaboration and client commitments. This policy is effective from January 1, 2026.

## Eligibility for Remote Work
Not all employees are eligible for remote work. The following rules apply:

- **Confirmed permanent employees** (post-probation) are eligible for remote work.
- **Probationers** (first 6 months of employment) must work from the office **5 days per week** with no remote work allowed.
- **Employees on Performance Improvement Plan (PIP)** must work from the office full-time until the PIP is successfully completed.
- **Employees with active disciplinary proceedings** are required to work from office.
- Client-facing roles may have additional in-office requirements as defined by the business head.

## Standard Hybrid Work Terms
For eligible employees, the standard hybrid work arrangement is:

- **Maximum 3 days per week** can be worked remotely.
- **Minimum 2 days per week in-office is mandatory** — typically Tuesday and Thursday, unless otherwise agreed with the manager.
- Remote work days are not fixed — they can vary week to week with manager's prior knowledge.
- Employees cannot carry over unused remote days to the following week.

## Core Working Hours
Core hours are mandatory for all employees regardless of work location.

- **Core hours: 10:00 AM - 5:00 PM IST** — employees must be available on Slack, Teams, and phone during these hours.
- **Flexible start window: 7:00 AM - 11:00 AM** — employees can choose their start time within this window.
- End time adjusts accordingly: if you start at 7 AM, you can finish by 4 PM; if you start at 11 AM, finish by 8 PM.
- All team meetings must be scheduled within core hours (10 AM - 5 PM IST) only.
- Employees working from home must be available for video calls and must not be unreachable during core hours.

## Remote Work Request and Approval Process
1. **Log the remote work request on the HRMS portal** at least **48 hours in advance**.
2. The manager has **24 hours** to approve or reject the request.
3. If no response within 24 hours, it is treated as approved by default.
4. **Recurring weekly schedule**: Employees who want a fixed remote day pattern (e.g., WFH every Monday and Wednesday) can apply for a **3-month recurring approval** via HRMS. Requires manager sign-off.
5. **100% remote (fully remote) arrangements**: Requires formal approval from both the manager and the HRBP. Reviewed every 6 months.
6. Ad-hoc remote requests on the day of are permitted for genuine emergencies with immediate manager intimation.

## Work From Abroad (WFA)
Working from outside India is a special arrangement with additional requirements.

- **Maximum 30 days per calendar year** can be worked from outside India.
- Requires approval from **HRBP + Finance** at least **2 weeks in advance**.
- Employees must ensure compliance with local tax and immigration laws of the country they are in.
- The company is not liable for any visa, work permit, or tax obligations that arise from Work From Abroad.
- WFA days count toward the 3-day-per-week remote work cap.
- Employees on WFA must still be available during IST core hours (10 AM - 5 PM IST).

## Infrastructure Requirements for Remote Work
To work effectively from home, employees must have:

- Stable broadband connection with a minimum of **50 Mbps download speed** (recommended).
- A dedicated, quiet workspace free from distractions during core hours.
- Company-issued laptop with up-to-date security patches and VPN installed.
- Webcam functional for video calls — video is mandatory for team stand-ups and 1:1s.
- Broadband reimbursement of Rs 1,500 per month is available (submit monthly bills via HRMS).

## Remote Work Policy Violations
Repeated violations of the remote work policy can lead to the remote work privilege being revoked.

Violations include:
- Being unreachable during core hours without prior notice.
- Not being available for scheduled meetings.
- Declining more than 3 ad-hoc manager requests in a month.
- Misusing WFA by working from unapproved locations.

Managers can escalate repeat violations to HRBP for review and potential suspension of remote work privileges.

## Special Situations

### Maternity or Medical Reasons
Employees returning from maternity leave may work fully from home for up to 3 months on mutual agreement with their manager and HRBP approval.

### Natural Disasters or Emergencies
In case of natural disasters, government directives, or pandemics, the company may mandate work from home for all employees. These situations override standard hybrid rules.

### New Joiners
New employees (within first 3 months) are encouraged to come to office as much as possible to build relationships and understand culture, even if technically eligible for remote work. Managers have discretion to require more in-office days during the onboarding period.

## Responsibilities While Working Remotely
- Maintain the same productivity and quality standards as in-office.
- Secure your home workspace — do not allow others to view company data or sit in on confidential calls.
- Use the company VPN at all times when accessing internal tools or data.
- Report any security incidents (lost laptop, data breach) immediately to IT helpdesk.

## Contact for Remote Work Queries
- HR Business Partner: hrbp@enterprise.com
- IT Helpdesk (VPN/security issues): ithelpdesk@enterprise.com (ext. 1234)
""",

"data/hr_docs/onboarding_guide.md": """# New Employee Onboarding Guide — Enterprise Corp

## Overview
Welcome to Enterprise Corp! This guide covers everything you need to do before joining, on your first day, during your first week, and across your first 90 days. Follow this checklist to ensure a smooth onboarding experience.

## Pre-Joining Checklist (7 Days Before Joining Date)
Complete the following steps before your first day:

- Accept the offer letter digitally on the HRMS portal. You will receive login credentials via email from HR within 2 days of offer acceptance.
- Submit KYC documents online via HRMS: Aadhaar card (front and back), PAN card, 3 passport-size photographs, bank account details (account number + IFSC code), and educational certificates.
- An IT asset request is auto-triggered once you accept the offer — your laptop will be ready on Day 1.
- You will receive a buddy assignment email from HR. Your buddy is a peer from your team who will help you get settled in the first month.
- Review the Employee Handbook sent to your personal email. Note the Code of Conduct, Data Privacy Policy, and IT Acceptable Use Policy.
- Download and install the Slack app on your personal phone. You will get workspace access on Day 1.
- If you have a current employer, confirm your last working day and provide the relieving letter or experience letter to HR (hrbp@enterprise.com) before or on joining.

## Day 1 Checklist
Your first day is structured to get you set up quickly and comfortably.

- **Arrive at Floor 2, Building A — IT Helpdesk** to collect your company laptop, access card, and welcome kit. This is open from 8:30 AM to 10:00 AM for new joiners.
- Attend the **HR Induction session: 9:30 AM - 12:30 PM** in Conference Room Atlas (Building B, Floor 3). Topics covered: company culture, HR policies overview, benefits enrollment, HRMS portal walkthrough.
- Set up your corporate email and verify Slack workspace access.
- Complete your payroll enrollment on HRMS: verify bank account, submit PAN, choose PF contribution preference.
- **Enroll in the company health insurance** on the benefits portal within 30 days of joining. After 30 days, enrollment is closed until the next annual window. This covers self, spouse, and up to 2 dependent children.
- Meet your reporting manager for a welcome 1:1 in the afternoon. Discuss initial priorities, team structure, and first-week schedule.
- Get your access card activated at the security desk on Floor 1 after the IT helpdesk visit.

## Week 1 Mandatory Training
All employees must complete the following mandatory training modules within the first 7 days of joining.

- **POSH Training** (Prevention of Sexual Harassment): Online module, approximately 2 hours. Completion certificate auto-generated on HRMS. Mandatory under POSH Act 2013.
- **Code of Conduct Training**: Covers conflict of interest, confidentiality, anti-bribery, and ethical behavior standards. Approximately 1.5 hours online.
- **Data Privacy and GDPR Training**: Covers handling of customer data, personal data, and data retention policies. Approximately 1 hour online.
- **IT Security Training**: Covers password policies, phishing awareness, VPN usage, and the Acceptable Use Policy. Approximately 1 hour online. Requires digital signature on AUP.
- Set up a weekly 1:1 cadence with your reporting manager. Weekly 1:1s are strongly recommended for the first 3 months.
- Attend your team standup or sprint planning meeting to begin understanding team workflows.
- Request access to key tools: Jira, Confluence, GitHub, and any role-specific applications via the IT helpdesk.

## 30-60-90 Day Development Plan

### First 30 Days — Shadow and Learn
- Shadow team members to understand processes, tools, and how work gets done.
- Complete all mandatory training modules.
- Set up introductory meetings with 10+ colleagues across your team and adjacent teams.
- Understand your OKRs (Objectives and Key Results) for the current quarter with your manager.
- Identify your development goals for the year and discuss with your manager.
- Complete the New Joiner Survey sent by HR at the end of Week 2.

### Days 31-60 — Ramp Up and Contribute
- Take on 1 small project or task with mentor support from your buddy or a senior team member.
- Draft your first OKR set with manager review and approval.
- Attend at least 2 internal Tech Talks or learning sessions.
- Begin using company tools independently (Jira, Confluence, Slack, HRMS).
- Complete the Probation Mid-Point check-in with your manager at the 60-day mark.

### Days 61-90 — Full Contribution
- Reach full independent contribution on your role responsibilities.
- Finalize your OKRs for the current quarter.
- Complete the 90-day review with your manager. Manager submits a probation assessment form to HR.
- If assessment is satisfactory, you will receive your probation confirmation letter.
- Begin exploring L&D budget options for certifications or courses relevant to your role.

## Key Documents Required for Full Onboarding
Provide the following to HR within 30 days of joining:
- Relieving letter or experience letter from previous employer.
- Last 3 months' salary slips from previous employer (for background verification).
- Educational degree certificates (Bachelor's, Master's, or equivalent).
- Address proof (Aadhaar or utility bill for current address).
- Any certifications mentioned in your resume.

## Key Contacts for New Joiners
- HR Business Partner: hrbp@enterprise.com
- IT Helpdesk (laptop, access, tools): ithelpdesk@enterprise.com (ext. 1234)
- Payroll queries: payroll@enterprise.com
- Employee Assistance Program (24x7 counselling): 1800-XXX-XXXX
- Facilities (seating, parking, cafeteria): facilities@enterprise.com

## HRMS Portal Access
- URL: https://hrms.enterprise.com
- Default password: Employee ID + Date of Birth in DDMMYYYY format
- Password must be changed on first login. Password policy: minimum 10 characters, 1 uppercase, 1 number, 1 special character.
- Multi-factor authentication (MFA) is mandatory. Set up the Authenticator app on your first login.

## FAQs for New Joiners

When will I receive my first salary?
Salary is credited on the last working day of the month. If you join after the 15th of the month, your first salary may be in the following month's cycle.

Can I work from home in my first month?
Probationers are required to be in office 5 days per week. Remote work is not available during probation (first 6 months).

When does my health insurance start?
Health insurance coverage begins from the date of joining. You must enroll within 30 days on the benefits portal to activate coverage.
""",

"data/hr_docs/grievance_compliance.md": """# Grievance Redressal & Compliance Policy — Enterprise Corp

## Overview
Enterprise Corp is committed to maintaining a fair, transparent, and respectful workplace. This policy establishes clear channels for employees to raise concerns, the timelines for resolution, and the standards of conduct expected from all employees. This policy is effective from January 1, 2026.

## Grievance Submission Channels
Employees can raise grievances through any of the following channels:

1. **HRMS Portal Grievance Module** (recommended): The ticket is automatically assigned to your HR Business Partner with an SLA tracker.
2. **Direct Email**: Write to grievance@enterprise.com. Include your employee ID, department, nature of grievance, and supporting documents.
3. **Anonymous Ethics Hotline**: Call 1800-XXX-1234 (toll-free, available 24x7). Callers can report issues anonymously. The hotline is managed by a third-party provider for complete confidentiality.
4. **Walk-in to HR**: Visit the HR desk (Floor 4, Building B) for sensitive matters that require face-to-face discussion.

All grievances are treated confidentially. Retaliation against any employee for raising a genuine grievance is a disciplinary offense.

## Grievance Resolution Timeline
Enterprise Corp follows a structured timeline to ensure all grievances are addressed promptly.

- **Acknowledgement**: Within 48 hours of grievance submission.
- **Initial response**: HRBP provides an initial response and investigation plan within 7 working days.
- **Resolution**: Standard grievances resolved within 21 working days.
- **Complex cases**: Cases involving multiple parties, legal implications, or investigations may take up to 45 working days.
- **Escalation**: If unresolved within 45 working days, the case is escalated to the Chief Human Resources Officer (CHRO) for direct intervention.
- **Appeal**: Employees may appeal a resolution decision to the CHRO within 10 working days of receiving the resolution.

## POSH Policy — Prevention of Sexual Harassment
Enterprise Corp has a **zero-tolerance policy** for sexual harassment in the workplace, as defined under the Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013.

- An **Internal Complaints Committee (ICC)** has been constituted per the POSH Act 2013 and includes an external independent member.
- All employees (permanent, contractual, and interns) must complete the **mandatory POSH training within 30 days of joining**.
- POSH training is conducted online via the HRMS portal and takes approximately 2 hours.
- **Complaints must be filed within 3 months of the incident** with the ICC.
- The ICC investigates all complaints confidentially and completes the inquiry within 90 days.
- Both the complainant and respondent are given an equal opportunity to present their case.
- ICC contact: icc@enterprise.com

## Code of Conduct
All employees are expected to uphold the highest standards of professional conduct.

- **Conflict of Interest**: Disclose any potential conflict of interest in writing to your HRBP within 14 days of the conflict arising.
- **Outside Employment**: No outside employment, consulting, or directorship is permitted without written HRBP approval.
- **Confidentiality**: Company information, client data, product roadmaps, and financial results are strictly confidential. Non-disclosure obligations continue for 2 years after leaving the company.
- **Social Media Policy**: Employees must not post confidential company information or disparaging content about the company on social media.
- **Anti-Bribery**: Gifts from vendors or clients above Rs 1,000 in value must be declared to HR. No cash gifts to be accepted under any circumstances.

## Disciplinary Process
When an employee violates company policies or code of conduct, the following structured process applies:

1. **Verbal Warning**: Manager issues a verbal warning, documented in HRMS. Employee is informed of the specific behavior that needs to change.
2. **Written Warning**: Formal letter issued by HR with clear improvement expectations. The employee is placed under 6-month monitoring.
3. **Performance Improvement Plan (PIP)**: A structured 90-day improvement plan with measurable targets. Employee must work from office full-time during PIP. Manager conducts fortnightly reviews.
4. **Suspension Pending Investigation**: For serious misconduct (fraud, harassment, data breach), employee may be placed on paid suspension while investigation is ongoing. Typically lasts 15-30 days.
5. **Termination**: With full due process, appropriate notice period (or pay in lieu), and a Full & Final settlement. Employees terminated for gross misconduct may not receive notice pay.

## Notice Period Policy
Notice period depends on employment band and probation status.

- **Probation period employees**: 1 month notice by either party (employee or company).
- **Confirmed employees Band 1, Band 2, Band 3**: 2 months notice period.
- **Senior employees Band 4 (Manager/Senior Manager), Band 5 (Director), Band 6 (VP/Senior Director)**: 3 months notice period.
- The company may ask an employee to serve the full notice period or may offer payment in lieu of notice (garden leave) at its discretion.
- Notice period buy-out: Employees may buy out their notice period by paying the equivalent salary to the company. HRBP approval required.

## Ethics and Anti-Corruption
- Enterprise Corp operates a zero-tolerance policy for bribery and corruption.
- Report any suspected fraud, bribery, or financial misconduct to the Ethics Hotline: 1800-XXX-1234 (anonymous).
- Ethics Email: ethics@enterprise.com (confidential, monitored by the Legal team).
- Whistleblower protection: Employees who report ethical violations in good faith are protected from retaliation under Enterprise Corp's Whistleblower Policy.

## Background Verification
- All employment offers are subject to successful background verification.
- Background checks cover: identity verification, education credentials, previous employment, and criminal record check (where legally permitted).
- If any discrepancy is found in documents or declarations, the offer or employment may be rescinded.
- Background verification is typically completed within 3 weeks of joining.

## Contact for Compliance Queries
- HRBP: hrbp@enterprise.com
- ICC (POSH complaints): icc@enterprise.com
- Ethics Hotline (anonymous): 1800-XXX-1234 (24x7)
- Legal Team: legal@enterprise.com
""",

"data/hr_docs/learning_development.md": """# Learning & Development Policy — Enterprise Corp 2026

## Overview
Enterprise Corp invests in continuous learning to help employees grow professionally and stay current in their fields. This policy covers the annual L&D budget, eligible expenses, the claim process, internal programs, and sponsorship rules. Effective from April 1, 2026.

## Annual L&D Budget
Every permanent employee receives a dedicated learning and development budget of **Rs 25,000 per year**.

- Budget is available from **April 1** each year and resets annually on April 1.
- **Unused L&D budget does NOT carry forward** to the next financial year. Any unused amount lapses on March 31.
- Probationers are eligible for the L&D budget from their date of joining (prorated for the remaining financial year).
- The budget can only be used for pre-approved learning activities.
- Band 4+ employees may request an additional Rs 10,000 top-up for leadership development programs, subject to HRBP approval.

## Eligible Expenses
The L&D budget can be used for the following categories of learning:

### Online Learning Platforms
- Coursera (individual and guided projects)
- Udemy Business
- LinkedIn Learning
- Pluralsight
- A Cloud Guru / Cloud Academy
- Codecademy Pro

### Professional Certifications
The following certifications are pre-approved and do not require individual HRBP sign-off:
- AWS certifications: Cloud Practitioner, Solutions Architect, Developer, SysOps, DevOps Engineer, Specialty exams
- Microsoft Azure certifications: AZ-900, AZ-104, AZ-204, AZ-305, DP-203, AI-900, AI-102
- Google Cloud certifications: Associate Cloud Engineer, Professional Cloud Architect, Data Engineer
- Project Management: PMP (PMI), PRINCE2, Agile/Scrum certifications (CSM, CSPO, SAFe)
- Finance certifications: CFA Level 1, FRM Part 1, CAIA
- Cybersecurity: CISA, CISSP, CEH, CompTIA Security+, OSCP
- Data and Analytics: Databricks certifications, Tableau Desktop Specialist, Power BI Data Analyst

### Books and Learning Materials
- Technical books (print or e-books) up to Rs 3,000 per book.
- Industry journals and subscriptions (annual, up to Rs 5,000).

### Conferences and Workshops
- Conference attendance (registration fees only, travel and accommodation are separate) with manager approval.
- Industry workshops and seminars.

### Coaching and Mentoring
- External executive coaching (Band 4+, pre-approved by HRBP).
- Domain-specific mentoring programs.

## L&D Budget Claim Process
Follow these steps to claim your L&D reimbursement:

1. **Pre-Approval**: Raise a pre-approval request in the **HRMS L&D module** before purchasing the course or registering for the certification. Do not purchase before getting pre-approval — unapproved expenses will not be reimbursed.
2. **Complete the Learning**: Attend the course, complete the modules, and clear the certification exam.
3. **Submit Expense Claim**: Upload the following documents in HRMS: payment receipt or invoice, course completion certificate or exam score report, and a brief description of how this learning applies to your current role.
4. **Approval**: Manager approves the claim within 7 working days. Finance processes the reimbursement.
5. **Reimbursement**: Amount is credited in the next payroll cycle after Finance approval.

If you fail a certification exam, you can claim the cost of one re-attempt within the same financial year, provided it is within your Rs 25,000 budget.

## Internal Learning Opportunities
Enterprise Corp offers several free internal programs in addition to the annual L&D budget:

- **Internal Tech Talks**: Every alternate Friday, 3:00 PM - 4:00 PM IST. Hosted by senior engineers, product managers, or external guest speakers. Recordings are available on Confluence within 48 hours.
- **Leadership Development Programme (LDP)**: Exclusive to Band 4 and above employees. A 12-week structured cohort program covering strategic thinking, people management, and executive communication. Run twice a year.
- **Mentoring Programme**: A 6-month one-on-one engagement pairing employees with a senior leader from a different function. Open to all confirmed employees. Applications open in April and October.
- **Quarterly Hackathon**: Cross-functional teams of 3-5 members solve real business problems over a weekend. Prizes up to Rs 1,00,000 for winning teams. All employees are eligible.
- **Learning Circles**: Peer-led learning groups focused on specific topics (e.g., Machine Learning, System Design, Product Strategy). Join or start a circle via Slack.
- **On-the-Job Assignments (Stretch Projects)**: Short-term cross-functional project opportunities for employees looking to expand skills.

## Full Sponsorship — Beyond Rs 25,000
For high-value certifications or programs that cost more than Rs 25,000, employees can apply for full company sponsorship.

- **Eligibility**: Minimum 1 year of service at Enterprise Corp.
- **Approval**: Requires manager + HRBP approval. Business case must demonstrate direct value to the role and the company.
- **Service Commitment**: A minimum of **1 year of continued employment** is required after completing the sponsored certification or program.
- **Clawback Policy**: If the employee resigns within **6 months** of completing the sponsored certification, **50% of the sponsorship amount** is recovered from the Full & Final settlement. If leaving within 3 months, 75% is recovered.
- **Eligible programs for full sponsorship**: MBA (part-time or executive), advanced cloud certifications, specialized data science programs, CFA Level 2 and 3.

## Learning Goals and OKRs
- Employees are encouraged to include 1-2 learning goals in their quarterly OKRs.
- Completion of a certified course or external certification earns a recognition badge on the company intranet.
- Employees who complete a certification should present their learnings in an Internal Tech Talk to share knowledge.

## Manager Responsibilities
Managers are expected to:
- Actively discuss L&D goals with each team member at least once per quarter.
- Approve or reject L&D pre-approval requests within 5 working days.
- Ensure team members are aware of the annual budget and internal programs.
- Nominate eligible team members for the Leadership Development Programme and Mentoring Programme.

## Contact for L&D Queries
- L&D Team: learning@enterprise.com
- HRBP: hrbp@enterprise.com
"""
}

for path, content in DOCS.items():
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Created: {path}")

# ── Structured HR Data Files ──────────────────────────────────────────────────
salary_bands = {"bands": [
    {"band":"B1","title":"Associate",          "min_ctc":400000, "max_ctc":700000,  "notice_period_months":1, "esop_eligible":False},
    {"band":"B2","title":"Senior Associate",   "min_ctc":700000, "max_ctc":1200000, "notice_period_months":2, "esop_eligible":False},
    {"band":"B3","title":"Lead / Senior Lead", "min_ctc":1200000,"max_ctc":2000000, "notice_period_months":2, "esop_eligible":False},
    {"band":"B4","title":"Manager / Sr Manager","min_ctc":2000000,"max_ctc":3500000,"notice_period_months":3, "esop_eligible":True},
    {"band":"B5","title":"Director",           "min_ctc":3500000,"max_ctc":6000000, "notice_period_months":3, "esop_eligible":True},
    {"band":"B6","title":"VP / Senior Director","min_ctc":6000000,"max_ctc":12000000,"notice_period_months":3,"esop_eligible":True},
]}
with open("data/hr_structured/salary_bands.json","w") as f:
    json.dump(salary_bands, f, indent=2)
print("  Created: data/hr_structured/salary_bands.json")

headcount_rows = [
    ["department","headcount","open_positions","attrition_ytd","avg_tenure_years"],
    ["Engineering","342","18","24","3.2"],
    ["Product","87","5","6","2.8"],
    ["Sales","145","12","19","2.1"],
    ["HR","34","2","3","4.1"],
    ["Finance","52","3","4","3.8"],
    ["Operations","78","6","8","2.9"],
    ["Marketing","41","3","5","2.5"],
    ["Legal & Compliance","18","1","1","5.2"],
]
with open("data/hr_structured/headcount.csv","w",newline="") as f:
    csv.writer(f).writerows(headcount_rows)
print("  Created: data/hr_structured/headcount.csv")

# placeholder .gitkeep files
for d in ["data/index","data/eval"]:
    open(f"{d}/.gitkeep","w").close()

print("\n  Sample data created successfully!")
print("  Next: python component_a_hr_indexing.py")
