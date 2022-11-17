

##
##  Packages.
import os, pandas, json


##
##  Class for report.
class report:

    def __init__(self, train=None, check=None, exam=None, test=None):

        self.train  = train
        self.check  = check
        self.exam   = exam
        self.test   = test
        pass
    
    def summarize(self):

        if(self.train):

            self.train = pandas.DataFrame(self.train)
            self.train.columns = ["train " + i for i in self.train.columns.tolist()]
            pass

        if(self.check):

            self.check = pandas.DataFrame(self.check)
            self.check.columns = ["check " + i for i in self.check.columns.tolist()]
            pass

        if(self.exam):

            self.exam = pandas.DataFrame(self.exam)
            self.exam.columns = ["exam " + i for i in self.exam.columns.tolist()]
            pass

        if(self.test):
            
            self.test  = pandas.DataFrame(self.test)
            self.test.columns  = ["test "  + i for i in self.test.columns.tolist() ]
            pass        

        self.summary = pandas.concat([self.train, self.check, self.exam, self.test], axis=1)
        pass

    def save(self, folder):

        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, 'report.csv')
        self.summary.to_csv(path, index=False)
        pass

##
##  The [write] function for saving the content to file.
# def write(content, folder, name):

#     with open(os.path.join(folder, name), "w+") as paper:
#         json.dump(content, paper)
#         pass


