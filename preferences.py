from sklearn.metrics.pairwise import cosine_similarity as cosine
from typing import List
import pandas as pd
import numpy as np
from numpy.linalg import norm
from tkinter import Tk, simpledialog, messagebox   # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import sys
import random
import math
# def cosine(A,B):
#     return np.dot(A,B)/(norm(A)*norm(B))
def preferences_func(csv, testing):
    df = pd.read_excel(csv)

    def data_cleaning_pipeline(df):
        # Renaming columns for saving time
        df = df.rename(columns={"""How dutch are you?
1. Have you lived in the Netherlands for 10+ years in total?2. Did you study in a highschool where Dutch was the language of instruction?3. Is Dutch your native language?4. Do y..."""
        :'Dutch/International','Would you like your teammates to be from a diverse background or not?':'Diversity?','What are your hobbies? (Choose 2-5 from this list that you are most interested in)':'List three hobbies/interests'})
        df['Diversity?'] = df['Diversity?'].fillna('NO')
        new_hobby_columns = set()
        ages = set()
        for index,row in df.iterrows():
            hobbies = row['List three hobbies/interests'].split(';')
            age = row['Age']
            if age == '>23':
                row['Age'] = 23
            # # INSERT PERSONALITY assignment here
            # df['extraversion'] = (5-row['...is reserved.']) + row['...is outgoing, sociable.']
            # df['agreeableness'] =(row['...is generally trusting.']) + (5-row['...tends to find fault with others.'])
            # df['conscientiousness'] =(row['...does a thorough job.']) + (5-row['...tends to be lazy.'])
            # df['neuroticism'] = (5-row['...is relaxed, handles stress well.']) + row['...gets nervous easily.']
            # df['openness'] = (5-row['...has few artistic interests.']) + row['...has an active imagination.']
            # Age group Encoder - DEPRECATED
            '''if age in list(range(21)):
            df.loc[index, 'Age 0-20'] = 1
            df['Age 0-20'] = df['Age 0-20'].fillna(0)
            ages.add('Age 0-20')
            elif age >=21 and age <=25:
            df.loc[index, 'Age 21-25'] = 1
            df['Age 21-25'] = df['Age 21-25'].fillna(0)
            ages.add('Age 21-25')
            elif age >= 26 and age <= 30:
            df.loc[index, 'Age 26-30'] = 1
            df['Age 26-30'] = df['Age 26-30'].fillna(0)
            ages.add('Age 26-30')
            else:
            df.loc[index, 'Age Old'] = 1
            df['Age Old'] = df['Age Old'].fillna(0)
            ages.add('Age 0-20')'''
            # Hobby encoding
            for i in hobbies:
                i2 = i.lower()
                df.loc[index,'hobbies.'+i2.strip()] = 1
                df['hobbies.'+i2.strip()] = df['hobbies.'+i2.strip()].fillna(0)
                new_hobby_columns.add('hobbies.'+i2.strip())
            # keep only necessary data
        return df[['Student Number', 'Dutch/International', 'Which region are you from?', 'Diversity?','Age']+list(new_hobby_columns) + list(ages)].set_index('Student Number')
        # StudentNumber, Origin, Origin Preference, Personality, Age
    df = data_cleaning_pipeline(df)
    # Remember to set categorical indices during matching
    # FOR TESTING ONLY
    if testing == True:
        def synthetic_dataset(n, df):
            '''FOR TESTING ONLY'''
            df_new = df
            for i in range(n):
                random_number = random.randrange(10000)
                random_dutch_int = random.choice(['Dutch', 'International'])
                random_region = random.choice(['Europe', 'Asia', 'North America', 'South America', 'Africa', 'Australia', 'Other'])
                # random_preference = random.choice(['Europe', 'Asia', 'North America', 'South America', 'Africa', 'Australia', 'Other'])
                random_preference = random.choice(['Yes','No'])

                hobbies = [i for i in df_new.columns.tolist() if i.startswith('hobbies.')]
                age = random.randrange(18,40)
                new_series = {'Age':age, 'Dutch/International':random_dutch_int, 'Which region are you from?':random_region, 'Diversity?':random_preference}

                for i in hobbies:
                    new_series[i] = random.choice([0.,1.])
                df_new = pd.concat([df_new, pd.DataFrame(new_series, index = [random_number])])

            return df_new.fillna(0)
        df = synthetic_dataset(14, df)
        # df.insert(loc = len(df.columns), column = 'hobbies.gaming', value = [1.])
        # df.insert(loc = len(df.columns), column = 'hobbies.piano', value = [0.])
        # df.iloc[0,2] = 'Asia'
        df.head()
    def similarity_finder(p1:pd.Series, p2:pd.Series):
        '''Returns simliarity score between two people'''

        # hyper parameters
        hobby_weight = 0.9
        region_weight = 0.3
        # Different Personalities attract
        personality_weight = -0.3
        dutch_weight = 0.5
        age_weight = 0.3
        age_drop_off_rate = 0.7
        all_features = set(p1.index.tolist()) | set(p2.index.tolist())
        similar_features = []
        for i in all_features:
            if p1[i] == p2[i]:
                similar_features.append(i)
        hobbies = [i for i in all_features if i.startswith('hobbies.')]
        # Replaced by Cosine similarity
        hobby_similarity = float(cosine(np.array(p1.loc[hobbies].tolist()).reshape(1, -1), np.array(p2.loc[hobbies].tolist()).reshape(1,-1)))
        # Age Diff
        age1 = p1.loc['Age']
        if age1 == '>23':
            age1 = 23
        age2 = p2.loc['Age']
        if age2 == '>23':
            age2 = 23
        age_diff = np.abs(int(age1) - int(age2))
        # personality_similarity = float(cosine(np.array(p1.loc[['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']]).reshape(1,-1),np.array(p2.loc[['extraversion', 'agreeableness','conscientiousness','neuroticism', 'openness']]).reshape(1,-1) ))
        if p2['Diversity?'] == True and p1['Diversity?'] == True and p1['Which region are you from?'] != p2['Which region are you from?']:
            region_compatibility = 1
        # elif p2['Where would you1 prefer your partner to be from? '] == p1['Which region are you from?'] or p1['Where would you prefer your partner to be from? '] == p2['Which region are you from?']:
        #     region_compatibility = 0.5
        else:
            region_compatibility = 0
        # Dutch Compatibility
        if (p1['Dutch/International'] == 'Dutch' and p2['Dutch/International'] == 'International') or(p1['Dutch/International'] == 'International' and p2['Dutch/International'] == 'Dutch'):
            dutch_compatibility = 1
        else:
            dutch_compatibility = 0
        compatibility_score = hobby_weight*hobby_similarity + region_weight*region_compatibility +  dutch_weight*dutch_compatibility + (np.exp(-age_drop_off_rate*abs(age_diff)))*age_weight
        return compatibility_score
    def preference_finder(df):
        '''Returns the preferences of each person with respect to other people'''
        # NVM Custom TTC Algorithm Maybe
        preferences = {}
        for index, row in df.iterrows():
            preferences[index] = {}
            for j in df.iterrows():
                if j[0] != index:
                    preferences[index][j[0]] = similarity_finder(row, j[1])
            preferences[index] = sorted(preferences[index], key = preferences[index].get, reverse = True)
        preferences = pd.DataFrame.from_dict(preferences, orient = 'index')
        #preferences.loc['null'] = ['null']*preferences.shape[1]
        return preferences
    print("Finished")
    return preference_finder(df)


class Matching:
    """Matching Class Stable Marriage"""
    def __init__(self, prefs: np.ndarray, group_size: int = 4, iter_count: int = 2, final_iter_count: int = 4):
        self.group_size = group_size
        self.prefs = prefs
        self.iter_count = iter_count
        self.final_iter_count = final_iter_count
        self.num_members = self.prefs.shape[0]
        self.num_groups = math.ceil(self.num_members / group_size)
        self.ungrouped = [i for i in range(self.num_members)]
        self.unfilled = []
        self.filled = []

        for i in random.sample(range(0, self.num_members), self.num_groups):
            self.unfilled.append(Group(self, [i]))
            self.ungrouped.remove(i)
        super().__init__()

    @staticmethod
    def from_csv(file_path, r: int = 4):
        prefs = np.genfromtxt(file_path, delimiter=',')
        return Matching(prefs, r)

    def get_mem_pref_for_group(self, mem: int, grp: List[int]) -> int:
        pref: int = 0
        for i in grp:
            pref += self.prefs[mem][i]
        pref = pref * (1.0 / len(grp))
        return pref

    def get_group_pref_for_mem(self, mem: int, grp: List[int]) -> int:
        pref: int = 0
        for i in grp:
            pref += self.prefs[i][mem]
        pref = pref * (1.0 / len(grp))
        return pref

    def get_group_score(self, y: List[int]) -> int:
        if len(y) <= 1:
            return 0
        score: int = 0
        for i in y:
            for j in y:
                if not (i == j):
                    score += self.prefs[i][j]
        score = score * (1.0 / (len(y) ** 2 - len(y)))
        return score

    def get_net_score(self) -> float:
        score = 0
        for i in self.filled:
            score += self.get_group_score(i.members)
        return score / self.num_groups

    def solve(self):
        while len(self.ungrouped) != 0:
            self.add_one_member()

        self.filled.extend(self.unfilled)
        self.unfilled = []
        self.optimize(use_filled=True)
        grps = []
        for i in self.filled:
            grps.append(i.members)
        return self.get_net_score(), grps

    def optimize(self, use_filled: bool = True):
        if use_filled:
            grps = self.filled
        else:
            grps = self.unfilled

        iters = self.final_iter_count if use_filled else self.iter_count

        for a in range(iters):
            for grp1 in grps:
                for mem1 in grp1.members:
                    for grp2 in grps:
                        if mem1 == -1:
                            break
                        if grp2 == grp1:
                            continue
                        for mem2 in grp2.members:
                            if mem1 == -1:
                                break
                            if mem2 == mem1:
                                continue
                            grp2mem1 = grp2.members.copy()
                            grp2mem1.remove(mem2)
                            grp2mem1.append(mem1)
                            grp1mem2 = grp1.members.copy()
                            grp1mem2.remove(mem1)
                            grp1mem2.append(mem2)

                            grp_one_new_score = self.get_group_score(grp1mem2)
                            grp_two_new_score = self.get_group_score(grp2mem1)

                            if (grp_one_new_score + grp_two_new_score > self.get_group_score(grp1.members) + self.get_group_score(
                                    grp2.members)):
                                grp1.add_member(mem2)
                                grp1.remove_member(mem1)
                                grp2.add_member(mem1)
                                grp2.remove_member(mem2)
                                mem1 = -1

    def add_one_member(self):

        proposed = np.zeros(
            shape=(len(self.ungrouped), len(self.unfilled)), dtype=bool)

        is_temp_grouped = [False for i in range(len(self.ungrouped))]

        temp_pref = np.zeros(
            shape=(len(self.ungrouped), len(self.unfilled)))

        temp_pref_order = np.zeros(
            shape=(len(self.ungrouped), len(self.unfilled)), dtype=int)

        for i, mem in enumerate(self.ungrouped):
            for j, grp in enumerate(self.unfilled):
                temp_pref[i][j] = self.get_mem_pref_for_group(mem, grp.members)

        for i, mem in enumerate(self.ungrouped):
            temp_pref_order[i] = np.argsort(temp_pref[i])[::-1]

        while is_temp_grouped.count(False) != 0:
            for i, mem in enumerate(self.ungrouped):

                if is_temp_grouped[i]:
                    continue

                if np.count_nonzero(proposed[i] == False) == 0:
                    is_temp_grouped[i] = True
                    continue
                for j in temp_pref_order[i]:
                    if proposed[i][j]:
                        continue

                    grp = self.unfilled[j]
                    proposed[i][j] = True
                    pref = self.get_group_pref_for_mem(mem, grp.members)
                    if pref > grp.tempScore:
                        if grp.tempMember >= 0:
                            is_temp_grouped[self.ungrouped.index(
                                grp.tempMember)] = False
                        grp.add_temp(mem)
                        is_temp_grouped[i] = True
                        break

        for grp in self.unfilled:
            if grp.tempMember < 0:
                continue
            self.ungrouped.remove(grp.tempMember)
            grp.add_permanently()

        self.optimize(use_filled=False)

        for grp in self.unfilled:
            if grp.size() >= self.group_size or len(self.ungrouped) == 0:
                self.filled.append(grp)

        for grp in self.filled:
            self.unfilled.remove(grp)


class Group:
    def __init__(self, game: Matching, members: List[int] = []):
        super().__init__()
        self.game = game
        self.members = members
        self.tempMember = -1
        self.tempScore = -1

    def add_member(self, x: int):
        self.members.append(x)

    def remove_member(self, x: int):
        self.members.remove(x)

    def add_temp(self, x: int) -> int:
        self.tempMember = x
        self.tempScore = self.game.get_group_pref_for_mem(x, self.members)
        return self.tempScore

    def add_permanently(self):
        if self.tempMember == -1:
            return
        self.add_member(self.tempMember)
        self.tempMember = -1
        self.tempScore = -1

    def size(self) -> int:
        return len(self.members)

def preference_format_converter(df):
    # Get all unique IDs from the data (both leaders and preferences)
    all_ids = pd.unique(df.values.ravel()).tolist()
    # prefs = df.to_dict('index')
    rankings = np.zeros(shape=(len(all_ids), len(all_ids)))
    for index, item in enumerate(all_ids):
        prefs = df.loc[item]
        prefs = prefs.tolist()
        for index2, item2 in enumerate(prefs):
            id_of_partner = all_ids.index(item2)
            score = len(all_ids) - index2-1
            rankings[index][id_of_partner] = score
    return rankings, all_ids

def return_finished_groups(groups, ids):
    final = []
    for i in groups:
        row = []
        for j in i:
            row.append(ids[j])
        final.append(row)
    return final

def write_to_xl(groups, tutorial_name = "Assignment Group"):
    df = pd.DataFrame(columns=['GroupSet','GroupName','SisId'])
    for grp_no, i in enumerate(groups):
        for j in i:
            row = {'GroupSet':tutorial_name, 'GroupName':tutorial_name+' Group '+ str(grp_no+1), 'SisId':j}
            # df = df.append(row, ignore_index=True)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    print(df)
    df.to_excel(tutorial_name+'.xlsx', index=False)

if __name__ == '__main__':
    # try:
    #     file = str(sys.argv[1])
    #     group_size = int(sys.argv[2])
    # except:
    #     print("Script not called correctly, add appropriate arguments(file_path group_size)")
    # Hyper Parameters
    # SET to false in production
    testing = True
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    tutorial_name = simpledialog.askstring("Input", "Please enter Tutorial Group Name:")

    file = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    if not file:
        messagebox.showerror("Error", "No file selected. Exiting.")

    group_size=5
    # file = "Matching Questionnaire (1-1).xlsx"
    prefs = preferences_func(file,testing)
    # print(prefs)
    # print(preference_format_converter(prefs))
    pref,students = preference_format_converter(prefs)

    n = 10

    # np.savetxt("Data.csv", pref, delimiter=',', fmt="%d")
    matching = Matching(pref, group_size=group_size, iter_count=2, final_iter_count=2)
    score, groups = matching.solve()
    # print(pref)
    # print(score)
    # print(groups)
    print(pref)
    print(students)
    groups = return_finished_groups(groups, students)
    print(groups)
    write_to_xl(groups, tutorial_name)
