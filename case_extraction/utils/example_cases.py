"""Script which contains some manually labelled cases for testing purposes."""
from pathlib import Path
from typing import List
from case_extraction.case import Case


def load_test_cases() -> List[Case]:
    """
    Load the test cases.

    :return: List of test cases.
    """
    case_1 = Case.manual_entry(
        filename="R_-v-_Christopher_James_McGee.pdf",
        offenses={"manslaughter"},
        defendants="christopher james mcgee",
        victims="georgia varley",
        premeditated="not premeditated",
        weapon="no weapon",
        vulnerable_victim="vulnerable",
        prior_convictions="no prior convictions",
        physical_abuse="no physical abuse",
        emotional_abuse="no emotional abuse",
        age_mitigating="not mitigate",
        race_aggrevating="race not an aggrevating factor",
        religious_aggrevating="religion not an aggrevating factor",
        offender_confession="no confession",
        victim_sex="female",
        victim_age="16",
        offender_age="",
        offender_sex="male",
        relationship="none",
        outcome={"5 years inprisonment"}
    )
    case_2 = Case.manual_entry(
        filename="Sentencing_remarks_of_His_Honour_Judge_Eccles_Q.C.:_R_-v-_Ben_Blakeley.pdf",
        offenses={"murder", "obstruction of justice"},
        defendants="ben blakeley",
        victims="jayden parkinson",
        premeditated="not premeditated",
        weapon="no weapon",
        vulnerable_victim="vulnerable",
        prior_convictions="prior convictions",
        physical_abuse="physical abuse",
        emotional_abuse="emotional abuse",
        age_mitigating="mitigate",
        race_aggrevating="race not an aggrevating factor",
        religious_aggrevating="religion not an aggrevating factor",
        offender_confession="no confession",
        victim_sex="female",
        victim_age="17",
        offender_age="22",
        offender_sex="male",
        relationship="partner",
        outcome={"life inprisonment", "minimum 20 years"}
    )
    case_3 = Case.manual_entry(
        filename="R_-v-_Pavlo_Lapshyn.pdf",
        offenses={"murder", "causing an explosion with an intent to endanger life",
                  "engaging in conduct in preparation of terrorist acts"},
        premeditated="premeditated",
        weapon="knife",
        vulnerable_victim="vulnerable",
        prior_convictions="no prior convictions",
        physical_abuse="no physical abuse",
        emotional_abuse="no emotional abuse",
        age_mitigating="mitigate",
        race_aggrevating="race an aggrevating factor",
        religious_aggrevating="religion an aggrevating factor",
        offender_confession="confession",
        victim_sex="male",
        victim_age="82",
        offender_age="25",
        offender_sex="male",
        relationship="none",
        victims={"mohammed saleem chaudhry", "mr saleem"},
        defendants={"pavlo lapshyn"},
        outcome={"40 years inprisonment", "life inprisonment"}
    )
    case_4 = Case.manual_entry(
        filename="R_-v-_David_Minto.pdf",
        offenses={"murder"},
        premeditated="premeditated",
        weapon="knife",
        vulnerable_victim="vulnerable",
        prior_convictions="no prior convictions",
        physical_abuse="physical abuse",
        emotional_abuse="no emotional abuse",
        age_mitigating="not mitigate",
        race_aggrevating="race not an aggrevating factor",
        religious_aggrevating="religion not an aggrevating factor",
        offender_confession="no confession",
        victim_sex="female",
        victim_age="16",
        offender_age="22",
        offender_sex="male",
        relationship="friend",
        victims={"sasha marsden"},
        defendants={"david minto"},
        outcome={"35 years", "life inprisonment"}
    )
    case_5 = Case.manual_entry(
        filename="R_-v-_Darrell_Desuze.pdf",
        offenses={"manslaughter", "violent disorder", "buglary"},
        premeditated="not premeditated",
        weapon="no weapon",
        vulnerable_victim="not vulnerable",
        prior_convictions="no prior convictions",
        physical_abuse="physical abuse",
        emotional_abuse="no emotional abuse",
        age_mitigating="mitigate",
        race_aggrevating="race not an aggrevating factor",
        religious_aggrevating="religion not an aggrevating factor",
        offender_confession="confession",
        victim_sex="male",
        victim_age="",
        offender_age="16",
        offender_sex="male",
        relationship="none",
        victims={"richard mannington bowes"},
        defendants={"darrell desuze"},
        outcome={"8 years detention"}
    )
    case_6 = Case.manual_entry(
        filename="R_-v-_James_McCormick.pdf",
        offenses={"fraud"},
        premeditated="premeditated",
        weapon="no weapon",
        vulnerable_victim="not vulnerable",
        prior_convictions="no prior convictions",
        physical_abuse="no physical abuse",
        emotional_abuse="no emotional abuse",
        age_mitigating="not mitigate",
        race_aggrevating="race not an aggrevating factor",
        religious_aggrevating="religion not an aggrevating factor",
        offender_confession="no confession",
        victim_sex="",
        victim_age="",
        offender_age="",
        offender_sex="male",
        relationship="none",
        victims={""},
        defendants={"james mccormick"},
        outcome={"10 years inprisonment"}
    )
    case_7 = Case.manual_entry(
        filename="R_-v-_Karl_Addo.pdf",
        offenses={"manslaughter"},
        premeditated="not premeditated",
        weapon="knife",
        vulnerable_victim="not vulnerable",
        prior_convictions="no prior convictions",
        physical_abuse="physical abuse",
        emotional_abuse="no emotional abuse",
        age_mitigating="not mitigate",
        race_aggrevating="race not an aggrevating factor",
        religious_aggrevating="religion not an aggrevating factor",
        offender_confession="confession",
        victim_sex="male",
        victim_age="23",
        offender_age="",
        offender_sex="male",
        relationship="none",
        victims={"sergio marquez"},
        defendants={"karl addo"},
        outcome={"six and a half years"}
    )
    case_8 = Case.manual_entry(
        filename="R_-v-_Kuntal_Patel.pdf",
        offenses={"acquiring a biological toxin"},
        premeditated="premeditated",
        weapon="no weapon",
        vulnerable_victim="not vulnerable",
        prior_convictions="no prior convictions",
        physical_abuse="no physical abuse",
        emotional_abuse="no emotional abuse",
        age_mitigating="not mitigate",
        race_aggrevating="race not an aggrevating factor",
        religious_aggrevating="religion not an aggrevating factor",
        offender_confession="confession",
        victim_sex="",
        victim_age="",
        offender_age="37",
        offender_sex="female",
        relationship="none",
        victims={},
        defendants={"kuntal patel"},
        outcome={"3 years"}
    )
    case_9 = Case.manual_entry(
        filename="R_-v-_Lowe.pdf",
        offenses={"murder"},
        premeditated="not premeditated",
        weapon="firearm",
        vulnerable_victim="not vulnerable",
        prior_convictions="no prior convictions",
        physical_abuse="no physical abuse",
        emotional_abuse="no emotional abuse",
        age_mitigating="mitigate",
        race_aggrevating="race not an aggrevating factor",
        religious_aggrevating="religion not an aggrevating factor",
        offender_confession="no confession",
        victim_sex="female",
        victim_age="66",
        offender_age="82",
        offender_sex="male",
        relationship="friend",
        victims={"christine lee", "lucy lee"},
        defendants={"john lowe"},
        outcome={"life inprisonment", "25 years"}
    )

    return [case_1, case_2, case_3, case_4, case_5, case_6, case_7, case_8, case_9]
