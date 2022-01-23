# GLUCOSE -> ASER format Preprocessing Replacement Dictionary
# These words below will be replaced as keys.
someone_a_list = ['someone_a\'s', 'someone\'s a\'s', 'someone_a']
someone_b_list = ['someone_b\'s', 'someone\'s b\'s', 'someone_b']
someone_c_list = ['someone_c\'s', 'someone\'s c\'s', 'someone_c']
someone_d_list = ['someone_d\'s', 'someone\'s d\'s', 'someone_d', 'someone\'s', 'someone']

single_group_list = ['some people_a', 'some people_b', 'some people_c', 'some people_d', 'some people',
                     'someone_a and someone_b', 'someone_b and someone_c', 'somepeople_a', 'somepeople_b',
                     'somepeople_c', 'somepeople']
single_groups_list = ['some people_a\'s', 'some people_b\'s', 'some people_c\'s', 'some people_d\'s',
                      'some people\'s a\'s', 'some people\'s b\'s', 'some people\'s c\'s', 'some people\'s d\'s',
                      'some people\'s', 'someone_a\'s and someone_b\'s', 'someone_b\'s and someone_c\'s',
                      'somepeople_a\'s', 'somepeople_b\'s', 'somepeople_c\'s', 'somepeople\'s']

single_thing_a_list = ['something_a\'s', 'something_a']
single_thing_b_list = ['something_b\'s', 'something_b']
single_thing_c_d_list = ['something_c\'s', 'something_d\'s', 'something\'s', 'something_c', 'something_d', 'something']

single_place_list = ['somewhere_a', 'somewhere_b', 'somewhere_c', 'somewhere_d', 'somewhere']
single_places_list = ['somewhere_a\'s', 'somewhere_b\'s', 'somewhere_c\'s', 'somewhere_d\'s', 'somewhere\'s']

# Words used as replacement value in GLUCOSE -> ASER, and will be the keys when ASER(GLUCOSE) -> ATOMIC
glucose_subject_list = ['i', 'you', 'he', 'she', 'someone', 'guy', 'man', 'woman', 'somebody']
glucose_object_list = ['it', 'that', 'something']
glucose_group_list = ['they', 'we']

# ASER(GLUCOSE) -> ATOMIC format replacement list, these words will be used as value to get back to ATOMIC format
ATOMIC_subject_list = ['PersonX', 'PersonY', 'PersonZ', 'PersonA', 'PersonB']
ATOMIC_object_list = ['ObjectX', 'ObjectY', 'ObjectZ']
ATOMIC_place_list = ['Somewhere']
ATOMIC_group_list = ['PeopleX', 'PeopleY']

# Unusable knowledge keywords list
Unusable = ['&', 'C', 'D', 'E', 'F', 'Someone_E', 'Someone_F', 'Someone_G', 'People_D', 'SomePeople_D', 'People_E',
            'SomePeople_E', 'People_F', 'SomePeople_F', 'Somewhere_C', 'Somewhere_D', 'Somewhere_E', 'Somewhere_F',
            'Something_D', 'Something_E', 'Something_F']

# GLUCOSE--ASER matching rule

# "out" means (GLUCOSE head -- relation -> ASER tail)
# "in" means (ASER tail -- relation -> GLUCOSE head)
# "both" means both mode above will work

# e.g. GLUCOSE Head A  > Cause/Enables >  ASER node
# 'out' can be 'Precedence' : Head A happens before ASER matching node B, so A cause/enables Node B
# 'in' can be 'Succession' : ASER matching node B happens after Head A, so A cause/enables Node B
# 'both' can be 'Conjunction' : HEAD A and ASER node B shall both happen

GLUCOSE_ASER_RULE = {
    "out": ["Precedence", "Result", ],
    "in": ["Succession", "Condition", "Reason", ],
    "both_dir": ["Synchronous", "Conjunction"],
}
