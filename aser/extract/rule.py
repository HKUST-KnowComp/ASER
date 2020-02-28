# CONNECTIVE_LIST = frozenset(['before', 'afterward', 'next', 'then', 'till', 'until', 'after', 'earlier', 'once', 'previously',
#                    'meantime', 'meanwhile', 'simultaneously', 'because', 'for', 'accordingly', 'consequently', 'so',
#                    'thus', 'therefore', 'if', 'but', 'conversely', 'however', 'nonetheless', 'although', 'and',
#                    'additionally', 'also', 'besides', 'further', 'furthermore', 'similarly', 'likewise', 'moreover',
#                    'plus', 'specifically', 'alternatively', 'or', 'otherwise', 'unless', 'instead', 'except'])

CLAUSE_WORDS = frozenset(['when', 'who', 'what', 'where', 'how', 'When', 'Who', 'What', 'Where', 'How', 'why', 'Why', 'which',
                'Which', '?'])

class Rule:
    def __init__(self, rules):
        if rules is None:
            self.positive_rules = list()
            self.negative_rules = list()
        else:
            self.positive_rules = rules['positive_rules']
            self.negative_rules = rules['negative_rules']

class EventualityRule(object):
    def __init__(self):
        self.positive_rules = list()
        self.possible_rules = list()
        self.negative_rules = list()

EVENTUALITY_PATTERNS = ['s-be-a', 's-v', 's-v-a', 's-v-a-X-o', 's-v-be-a', 's-v-be-o', 's-v-o', 's-v-o-o', 's-v-o-X-o',
                       's-v-v', 's-v-v-o', 's-v-o-v-o', 's-v-o-v-o-X-o', 's-v-X-o', 'spass-v', 'spass-v-X-o']

ALL_EVENTUALITY_RULES = dict()

A_rule = EventualityRule()
A_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
A_rule.positive_rules.append(('V1', 'dobj', 'O1'))
A_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt', 'NA'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('O1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/aux/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('S1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
A_rule.negative_rules.append(('O1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['s-v-o'] = A_rule

A_rule = EventualityRule()
A_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
A_rule.positive_rules.append(('V1', 'dobj', 'O1'))
A_rule.positive_rules.append(('V1', 'xcomp', 'V2'))
A_rule.positive_rules.append(('V2', 'dobj', 'O2'))
A_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt', 'NA'))
A_rule.possible_rules.append(('V2', '+advmod/neg/aux/compound:prt', 'NA'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('O1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('O2', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.negative_rules.append(('V1',
                              """-ccomp/nsubj:xsubj/mark/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('V2',
                              """-ccomp/nsubj:xsubj/mark/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('S1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
A_rule.negative_rules.append(('O1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
A_rule.negative_rules.append(('O2', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['s-v-o-v-o'] = A_rule

A_rule = EventualityRule()
A_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
A_rule.positive_rules.append(('V1', 'dobj', 'O1'))
A_rule.positive_rules.append(('V1', 'xcomp', 'V2'))
A_rule.positive_rules.append(('V2', 'dobj', 'O2'))
A_rule.positive_rules.append(('V2',
                              '+nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about',
                              'O3'))
A_rule.positive_rules.append(('O3', 'case', 'P1'))
A_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt', 'NA'))
A_rule.possible_rules.append(('V2', '+advmod/neg/aux/compound:prt', 'NA'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('O1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('O2', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('O3', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.negative_rules.append(('V1',
                              """-ccomp/nsubj:xsubj/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('V2',
                              """-ccomp/nsubj:xsubj/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('S1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
A_rule.negative_rules.append(('O1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
A_rule.negative_rules.append(('O2', """+nmod:of/nmod:for/nmod:at""", 'NA'))
A_rule.negative_rules.append(('O3', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['s-v-o-v-o-X-o'] = A_rule

A_rule = EventualityRule()
A_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
A_rule.positive_rules.append(('V1', 'dobj', 'O1'))
A_rule.positive_rules.append(('V1', 'iobj', 'O2'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('O1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('O2', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt', 'NA'))
A_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('O1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
A_rule.negative_rules.append(('O2', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['s-v-o-o'] = A_rule

A_rule = EventualityRule()
A_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case/compound:prt', 'NA1'))
A_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt/mark', 'NA2'))
A_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA3'))
A_rule.negative_rules.append(('S1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['s-v'] = A_rule

A_rule = EventualityRule()
A_rule.positive_rules.append(('V1', 'nsubjpass', 'S1'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('V1', '+advmod/neg/aux/auxpass/compound:prt/mark', 'NA'))
A_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('S1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['spass-v'] = A_rule

A_rule = EventualityRule()
A_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
A_rule.positive_rules.append(('V1', 'xcomp', 'V2'))
A_rule.positive_rules.append(('V2', 'dobj', 'O1'))
A_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt/mark', 'NA'))
A_rule.possible_rules.append(('V2', '+advmod/neg/aux/compound:prt/mark', 'NA'))
A_rule.possible_rules.append(('O1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('V2',
                              """-ccomp/parataxis/nsubj:xsubj/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('V0', '^ccomp', 'V1'))
A_rule.negative_rules.append(('V0', '^ccomp', 'V2'))
A_rule.negative_rules.append(('S1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
A_rule.negative_rules.append(('O1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['s-v-v-o'] = A_rule

A_rule = EventualityRule()
A_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
A_rule.positive_rules.append(('V1', 'xcomp', 'A1'))
A_rule.positive_rules.append(('A1', 'cop', 'V2'))
A_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt/mark', 'NA'))
A_rule.possible_rules.append(('A1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('S1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
A_rule.negative_rules.append(('A1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['s-v-be-a'] = A_rule

A_rule = EventualityRule()
A_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
A_rule.positive_rules.append(('V1', 'xcomp', 'V2'))
A_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt/mark', 'NA'))
A_rule.possible_rules.append(('V2', '+advmod/neg/aux/compound:prt/mark', 'NA'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('V2',
                              """-ccomp/parataxis/nsubj:xsubj/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('S1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['s-v-v'] = A_rule

A_rule = EventualityRule()
A_rule.positive_rules.append(('A1', '^cop', 'V1'))
A_rule.positive_rules.append(('A1', 'nsubj', 'S1'))
A_rule.possible_rules.append(('A1', '+advmod/neg/aux/compound:prt/det/amod/compound/nmod:poss/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.negative_rules.append(('A1',
                              """-ccomp/parataxis/mark/nmod:npmod/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nmod:after/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('S1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['s-be-a'] = A_rule

A_rule = EventualityRule()
A_rule.positive_rules.append(('A1', '^cop', 'V1'))
A_rule.positive_rules.append(('A1', 'nsubj', 'S1'))
A_rule.positive_rules.append(('A1',
                              '+nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about',
                              'O1'))
A_rule.positive_rules.append(('O1', 'case', 'P1'))
A_rule.possible_rules.append(('A1', '+advmod/neg/aux/compound:prt/amod/mark', 'NA'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('O1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.negative_rules.append(('A1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('S1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['s-v-a-X-o'] = A_rule

A_rule = EventualityRule()
A_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
A_rule.positive_rules.append(('V1',
                              '+nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about',
                              'O1'))
A_rule.positive_rules.append(('O1', 'case', 'P1'))
A_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt/mark', 'NA'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('O1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/nsubj:xsubj/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('S1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
A_rule.negative_rules.append(('O1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['s-v-X-o'] = A_rule

A_rule = EventualityRule()
A_rule.positive_rules.append(('V1', 'nsubjpass', 'S1'))
A_rule.positive_rules.append(('V1',
                              '+nmod:agent/nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about',
                              'O1'))
A_rule.positive_rules.append(('O1', 'case', 'P1'))
A_rule.possible_rules.append(('V1', '+advmod/neg/aux/auxpass/compound:prt/mark', 'NA'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('O1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/auxpass/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('S1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
A_rule.negative_rules.append(('O1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['spass-v-X-o'] = A_rule

A_rule = EventualityRule()
A_rule.positive_rules.append(('V1', 'nsubj', 'S1'))
A_rule.positive_rules.append(('V1', 'dobj', 'O1'))
A_rule.positive_rules.append(('V1',
                              '+nmod:into/nmod:for/nmod:around/nmod:with/nmod:poss/nmod:inside/nmod:at/nmod:outside_of/nmod:than/nmod:from/nmod:in/nmod:on/nmod:to/nmod:away_from/amod:as/nmod:down/nmod:up/nmod:tmod/nmod:along/nmod:over/nmod:out_of/nmod:of/nmod:without/nmod:by/nmod:through/nmod:about',
                              'O2'))
A_rule.positive_rules.append(('O2', 'case', 'P1'))
A_rule.possible_rules.append(('V1', '+advmod/neg/aux/compound:prt/mark', 'NA'))
A_rule.possible_rules.append(('S1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('O1', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.possible_rules.append(('O2', '+amod/neg/nummod/compound/det/nmod:poss/mark/det:qmod/case', 'NA'))
A_rule.negative_rules.append(('V1',
                              """-ccomp/parataxis/conj:but/advcl/dep/cc/punct/mark/conj:and/nmod:tmod/advcl:to/advcl:though/advcl:after/advcl:if/advcl:while/advcl:as/advcl:for/advcl:in/advcl:since/advcl:from/advcl:before/advcl:because/advcl:based_on/advcl:with/advcl:although/advcl:by/advcl:so/advcl:at/advcl:on/advcl:upon/advcl:until/advcl:'s/advcl:instead_of/advcl:despite/advcl:through/advcl:unless/advcl:in_order/advcl:ago""",
                              'NA'))
A_rule.negative_rules.append(('S1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
A_rule.negative_rules.append(('O1', """+nmod:of/nmod:for/nmod:at""", 'NA'))
A_rule.negative_rules.append(('O2', """+nmod:of/nmod:for/nmod:at""", 'NA'))
ALL_EVENTUALITY_RULES['s-v-o-X-o'] = A_rule


SEED_CONNECTIVE_DICT = {
    'Precedence': [['before']],
    'Succession': [['after']],
    'Synchronous': [['meanwhile'], ['at', 'the', 'same', 'time']],
    'Reason': [['because']],
    'Result': [['so'], ['thus'], ['therefore']],
    'Condition': [['if']],
    'Contrast': [['but'], ['however']],
    'Concession': [['although']],
    'Conjunction': [['and'], ['also']],
    'Instantiation': [['for', 'example'], ['for', 'instance']],
    'Restatement': [['in', 'other', 'words']],
    'Alternative': [['or'], ['unless']],
    'ChosenAlternative': [['instead']],
    'Exception': [['except']],
    'Co_Occurrence': list()
}