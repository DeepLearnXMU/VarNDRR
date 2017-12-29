#! /usr/bin/python

import sys
import cPickle

if len(sys.argv) != 6 and len(sys.argv) != 8:
    print '%s lbl_size prefix arg1 file, arg2 file, label file, [arg1.vcb], [arg2.vcb]'%sys.argv[0]
    sys.exit(1)

if len(sys.argv) == 6:
    arg1_vcb_file = None
    arg2_vcb_file = None
    lbl_size,prefix,arg1_file,arg2_file,lbl_file = sys.argv[1:]
else:
    lbl_size,prefix,arg1_file,arg2_file,lbl_file,arg1_vcb_file,arg2_vcb_file = sys.argv[1:]

def word_count(file_name):
    word_cnter = dict()
    
    for line in file(file_name, 'rU'):
        words = line.strip().split()
        for word in words:
            if word not in word_cnter:
                word_cnter[word] = 0
            word_cnter[word] += 1
    word_cnter = sorted(word_cnter.items(), lambda x,y:cmp(x[1], y[1]), reverse=True)

    vocab = dict()
    id_cnter = 1
    for word, _ in word_cnter:
        vocab[word] = id_cnter
        id_cnter += 1
    return vocab

if arg1_vcb_file == None and arg2_vcb_file == None:
    arg1_vcb = word_count(arg1_file)
    arg2_vcb = word_count(arg2_file)
else:
    arg1_vcb = cPickle.load(file(arg1_vcb_file, 'r'))
    arg2_vcb = cPickle.load(file(arg2_vcb_file, 'r'))

lbl_vcb = dict()
for l in xrange(int(lbl_size)):
    lbl_vcb['%s'%l] = l

# save vocabularies
if not arg1_vcb_file:
    cPickle.dump(arg1_vcb, file('arg1.vcb.pkl', 'w'))
    cPickle.dump(arg2_vcb, file('arg2.vcb.pkl', 'w'))

def file_idcvt(file_name, vcb):
    datas = []

    for line in file(file_name, 'rU'):
        words = line.strip().split()

        data = [vcb[word] if word in vcb else 0 for word in words] # Note, 0 means OOV
        datas.append(data)
    return datas

arg1_dta = file_idcvt(arg1_file, arg1_vcb)
arg2_dta = file_idcvt(arg2_file, arg2_vcb)
lbl_dta = file_idcvt(lbl_file, lbl_vcb)

# save datas
cPickle.dump(arg1_dta, file('%s.arg1.dta.pkl'%prefix, 'w'))
cPickle.dump(arg2_dta, file('%s.arg2.dta.pkl'%prefix, 'w'))
cPickle.dump(lbl_dta, file('%s.lbl.dta.pkl'%prefix, 'w'))
