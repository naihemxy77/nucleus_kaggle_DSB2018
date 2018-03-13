import numpy as np
import pandas as pd
import submission_encoding
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages

##Generate Test Masks and Submission files given pickle outputs from models
##In current file, if image type is not histological, then test masks will be
##predicted by the linear combination of otsu,iso and li.
submissionfilename = 'baseline_submission0312'
test_label = pickle.load(open( "Test_Label.p","rb" ))
test_label = pd.DataFrame(test_label, columns=['ImageId','ImageLabel'])

r_n = 10
c_n = 7
fig, m_axs = plt.subplots(r_n, c_n,figsize=(10,15))
for i in range(r_n):
    for j in range(c_n):
        id_pointer = c_n*i+j
        if id_pointer<test_label.shape[0]:
            m_axs[i][j].imshow(test_label.loc[id_pointer,'ImageLabel'])
            m_axs[i][j].axis('off')
            m_axs[i][j].set_title(str(id_pointer),fontsize=7)
pp = PdfPages(submissionfilename+'.pdf')
pp.savefig(fig)
pp.close()

submission_encoding.submission_gen(test_label, submissionfilename)