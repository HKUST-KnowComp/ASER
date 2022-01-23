# Variables and rules
PP_SINGLE = ['i', 'you', 'he', 'she', 'someone', 'guy', 'man', 'woman', 'somebody']
# SUBJS = ["person", "man", "woman", 
#         "someone", "somebody", "i", "he", "she", "you"]
O_SUBJS = ["i", "you", "he", "she"]
ATOMIC_SUBJS = ["PersonX", "PersonY", "PersonZ"] 
SUBJ2POSS = {"i":"my", "he": "his", "she":"her", "you":"your"}

stative_rules = {
    "in":[], "out":[],
    "both_dir":["Synchronous", "Reason", "Result", "Condition", 
        "Conjunction", "Restatement", "Alternative"]
}
cause_rules = {
    "out":["Succession", "Condition", "Reason", ],
    "in":["Precedence", "Result",], 
    "both_dir":["Synchronous", "Conjunction"], 
}
effect_rules = {
    "out":["Precedence", "Result",], 
    "in":["Succession", "Condition", "Reason",],
    "both_dir":["Synchronous", "Conjunction"], 
}

ASER_rules_dict = {
    "stative": stative_rules,
    "cause": cause_rules,  
    "effect": effect_rules,
}


def get_ppn_substitute_dict_head(head_split):
  atomic_head_pp_list = []
  for token in head_split:
    if token in PP_SINGLE:
      if not token in atomic_head_pp_list:
        atomic_head_pp_list.append(token)
  head_pp2atomic_pp = {}
  cnt = 0
  for pp in atomic_head_pp_list:
    head_pp2atomic_pp[pp] = ATOMIC_SUBJS[cnt]
    cnt += 1
    if cnt >= len(ATOMIC_SUBJS):
      break
  return head_pp2atomic_pp


def get_ppn_substitue_dict(head_split, tail_split):
  """
      input (list): the split result of a head
      
      output: a dict tha maps personal pronouns in 
              head_split to subjects in ATOMIC_SUBJS
  """
  
  head_subj = head_split[0]
  tail_subj = tail_split[0]
  
  if head_subj == tail_subj:
    if head_subj in PP_SINGLE:
      return get_ppn_substitute_dict_head(head_split + tail_split)
    else:
      # neither in PP_SINGLE
      return {}
  else:
    # head_subj != tail_subj
    if head_subj in PP_SINGLE and tail_subj not in PP_SINGLE:
      return get_ppn_substitute_dict_head(head_split + tail_split)
    elif head_subj not in PP_SINGLE and tail_subj not in PP_SINGLE:
      return {}
    elif head_subj in PP_SINGLE and tail_subj in PP_SINGLE:
      head_pp2atomic_pp = get_ppn_substitute_dict_head(head_split + tail_split)
      pp_list = [[key for key, atomic_subj in head_pp2atomic_pp.items() if atomic_subj == ATOMIC_SUBJS[i]][0]\
                 for i in range(len(head_pp2atomic_pp))
                ]
      num_subj = len(head_pp2atomic_pp)
      new_pp_dict = {}
      new_pp_dict[head_subj] = ATOMIC_SUBJS[0]
      new_pp_dict[tail_subj] = ATOMIC_SUBJS[1]
      pp_list.remove(head_subj)
      if tail_subj in pp_list:
        pp_list.remove(tail_subj)
      if len(pp_list) > 0:
        new_pp_dict[pp_list[0]] = ATOMIC_SUBJS[2]
      return new_pp_dict
    elif head_subj not in PP_SINGLE and tail_subj in PP_SINGLE:
      return get_ppn_substitute_dict_head(head_split + tail_split)

def filter_event(event):
  """
      Function of filtering eventualities
      input (str): the string of eventuality
      
      output: whether to filter it out or not.
  """
  tokens = event.split()
#   if tokens[-1] in SUBJS and tokens[-2] == "tell":
#     return True
#   if tokens[-1] in ["know", "say", "think"]:
#     return True
  # filter eventualities with only 2 tokens
  if len(tokens) <= 2:
    return True
  # filter hot verbs
  if any(kw in tokens for kw in ["say", "do", "know", "tell", "think", ]):
    return True
  # filter out errors that potentially due to the errors of the parser
  if tokens[0] in ["who", "what", "when", "where", "how", "why", "which", "whom", "whose"]:
    return True
  return False  