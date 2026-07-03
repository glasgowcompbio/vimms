import csv
import glob
import os
import re

import requests
from molmass import Formula




PROTON_MASS = 1.00727645199076
ELECTRON_MASS = Formula('H').isotope.mass - PROTON_MASS


class Adduct(object):
    def __init__(self,name):
        self.name = name
        self.synonyms = {}
    
    def add_synonym(self,dialect,name):
        self.synonyms[dialect] = name

    def __str__(self):
        return self.name + "  (" + ", ".join(["{}:{}".format(k,v) for k,v in self.synonyms.items()]) + ")"

class AdductThesaurus(object):
    def __init__(self):
        self.adduct_dict = {}
    def add_adduct(self,adduct):
        self.adduct_dict[adduct.name] = adduct
    def get_standard_name(self,name,dialect):
        for main_name,adduct in self.adduct_dict.items():
            local_name = adduct.synonyms.get(dialect,None)
            if local_name == name:
                return main_name
        return None



class AbstractAdductTransformer(object):
    def mass2ion(self,mass,adduct_name):
        raise NotImplementedError()

    def ion2mass(self,ion_mass,adduct_name):
        raise NotImplementedError()

    def adduct_list(self,mode = 'both'):
        raise NotImplementedError()

class AdductTransformer(AbstractAdductTransformer):
    def __init__(self,local_file_folder = None):
        self.adduct_rules = self._load_adduct_files(local_file_folder = local_file_folder)

    def adduct_list(self,mode = 'both'):
        if mode == 'positive':
            return [a for a,v in self.adduct_rules.items() if v['charge'] > 0]
        elif mode == 'negative':
            return [a for a,v in self.adduct_rules.items() if v['charge'] < 0]
        else:
            return list(self.adduct_rules.keys())

    def mass2ion(self,mass,adduct_name):
        try:
            mass_add = self.adduct_rules[adduct_name]['mass_add']
            mass_multi = self.adduct_rules[adduct_name]['mass_multi']
        except:
            print("{} not a valid adduct, trying to parse".format(adduct_name))
            pa = ParsingAdductTransformer()
            return pa.mass2ion(mass,adduct_name)
            # return None
        return mass*mass_multi + mass_add

    def ion2mass(self,ion_mass,adduct_name):
        try:
            mass_add = self.adduct_rules[adduct_name]['mass_add']
            mass_multi = self.adduct_rules[adduct_name]['mass_multi']
        except:
            print("{} not a valid adduct, trying to parse".format(adduct_name))
            pa = ParsingAdductTransformer()
            return pa.ion2mass(ion_mass,adduct_name)
            # return None
        return (ion_mass - mass_add)/mass_multi

    def _load_adduct_files(self,local_file_folder = None,adduct_rules = {}):
        file_urls = ['https://raw.githubusercontent.com/michaelwitting/adductDefinitions/master/adducts_pos.txt','https://raw.githubusercontent.com/michaelwitting/adductDefinitions/master/adducts_neg.txt']
        if local_file_folder:
            files = glob.glob(os.path.join(local_file_folder,'*.txt'))
        else:
            files = None
        if not files is None:
            for file_name in files:
                with open(file_name,'r') as f:
                    reader = csv.reader(f)
                    self._parse(reader,adduct_rules)
        else:
            for file_url in file_urls:
                r = requests.get(file_url)
                decoded_content = r.content.decode('utf-8')
                reader = csv.reader(decoded_content.splitlines(),delimiter ='\t')
                self._parse(reader,adduct_rules)
        return adduct_rules


    def _parse(self,csv_reader,adduct_rules):
        heads = next(csv_reader)
        for line in csv_reader:
            adduct_transformation = line[0]
            charge = int(line[1])
            mass_add = float(line[4])
            mass_multi = float(line[5])
            adduct_rules[adduct_transformation] = {'mass_add':mass_add,'mass_multi':mass_multi,'charge':charge}
        

class ParsingAdductTransformer(AbstractAdductTransformer):
    def __init__(self):
        self.adduct_thes = AdductThesaurus()
        self._load_adducts()
    

    def mass2ion(self,mass,adduct_name,dialect = None):
        if not dialect is None:
            adduct_string = self.adduct_thes.get_standard_name(adduct_name,dialect)
            if adduct_string is None:
                return None
        else:
            adduct_string = adduct_name

        params = adduct_string_parser(adduct_string)
        if params:
            return (mass*params[0] + params[1])/abs(params[2])
        else:
            return None
    
    def ion2mass(self,mass,adduct_name,dialect = None):
        if not dialect is None:
            adduct_string = self.adduct_thes.get_standard_name(adduct_name,dialect)
            if adduct_string is None:
                return None
        else:
            adduct_string = adduct_name
        params = adduct_string_parser(adduct_string)
        if params:
            return (mass*abs(params[2]) - params[1])/params[0] # check this!
        else:
            return None
    
    def get_transform_list(self):
        return list(self.adduct_thes.keys())

    def _load_adducts(self,positive_filename = 'adduct_csv_files/Adduct definitions and synonyms - positive adducts.csv',
                           negative_filename = 'adduct_csv_files/Adduct definitions and synonyms - negative adducts.csv'):
        try:
            self._parse_csv(positive_filename)
            self._parse_csv(negative_filename)
        except:
            print("Failed to load .csv files. Assuming normal dialect.")



    def _parse_csv(self,filename):
        import csv


        with open(filename,'r') as f:
            reader = csv.reader(f)
            top_heads = next(reader)
            main_heads = next(reader)
            dialect_pos = range(5,17)
            for line in reader:
                if len(line[0]) == 0:
                    continue # no main name
                new_adduct = Adduct(line[0])
                for dpos in dialect_pos:
                    if len(line[dpos]) > 0:
                        new_adduct.add_synonym(main_heads[dpos],line[dpos])
                self.adduct_thes.add_adduct(new_adduct)
        
        

    
# obsolete methods - for deletion

def get_transform_list():
    return AdductTransformer().adduct_list()

def get_positive_transform_list():
    return AdductTransformer().adduct_list(mode='positive')

def get_negative_transform_list():
    return AdductTransformer().adduct_list(mode='negative')



def adduct_string_parser(adduct_string):
    # Step 1, access the charge
    
    # initial hacky regex check
    hit = re.search('^\[([1-9][0-9]*)?M.*\]([1-9][0-9]*)?[+-]$',adduct_string)
    if not hit:
        print("String didn't conform to expected format")
        return None

    tokens = adduct_string.split(']')
    adduct_info = tokens[0]
    charge = tokens[1]

    if charge == '+':
        charge = 1
    elif charge == '-':
        charge = -1
    elif '+' in charge:
        charge = charge.replace('+','')
        charge = int(charge)
    elif '-' in charge:
        charge = charge.replace('-','')
        charge = -int(charge)
    else:
        charge = int(charge)

    
    tokens = adduct_info.split('M')
    m_info = tokens[0]
    adduct_info = tokens[1]
    
    m_info = m_info[1:] # remove the '['
    try:
        mass_multiplier = int(m_info)
    except:
        mass_multiplier = 1
    
    plus_pos = [('+',m.start()) for m in re.finditer("\+",adduct_info)]
    minus_pos = [('-',m.start()) for m in re.finditer("\-",adduct_info)]
    
    delim_positions = sorted(plus_pos + minus_pos,key = lambda x: x[1])
    


    mass_shift = 0
    for i,(t,d) in enumerate(delim_positions):
        if i == len(delim_positions) - 1:
            end_pos = len(adduct_info)
        else:
            end_pos = delim_positions[i+1][1]
        adduct = adduct_info[d+1:end_pos]

        # check for a pre-multiplier 
        # e.g. [M+2H]2+
        # we need to add the H twice.
        # Using Formula('2H') already removes an electron
        r = re.search('^[0-9]+',adduct)
        if not r is None:
            # get the multiplier
            multiplier = int(r.group())
            # extract the remaining adduct
            adduct = adduct[r.span()[1]:]
        else:
            multiplier = 1

        # get the mass
        formula = Formula(adduct)        
        adduct_mass = formula.isotope.mass

        # subtract if necessary
        if t == '-':
            adduct_mass = -adduct_mass

        # cumulative mass shift
        mass_shift += multiplier * adduct_mass
    
    # remove / add enough electrons
    mass_shift -= charge*ELECTRON_MASS

    return (mass_multiplier,mass_shift,charge)    





if __name__ == '__main__':
    # print("E mass: ",ELECTRON_MASS)

    # a_list = get_positive_transform_list()
    # print("Positive transformations:")
    # for a in a_list:
    #     print(a)

    # a_list = get_negative_transform_list()
    # print("Negative transformations:")
    # for a in a_list:
    #     print(a)

    # for adduct in adduct_rules:
    #     print()
    #     print("Transform 100 with {}: {}".format(adduct,mass2ion(100,adduct)))
    #     print("Transform back: ",ion2mass(mass2ion(100,adduct),adduct))


    # for adduct in adduct_rules:
    #     print()
    #     print(adduct)
    #     print(adduct_string_parser(adduct))
    #     print(adduct_rules[adduct])

    # adduct_thes = parse_csv()

    # print(adduct_thes.get_standard_name('(M+ACN+H)+','Waters'))

    at = ParsingAdductTransformer()
    print(at.mass2ion(100,'[M+2H]2+'))
    print(at.mass2ion(100,'[M-2H]2-'))
    print(at.mass2ion(120,'(M+ACN+H)+',dialect='Waters'))
    print(at.mass2ion(120,'(M+ACN+H)+')) # throws error
    print(at.mass2ion(130,'[M-2H]2-'))
    print(at.ion2mass(at.mass2ion(130,'[M-2H]2-'),'[M-2H]2-'))
    
    at2 = AdductTransformer()
    print(at2.mass2ion(100,'[M+2H]2+'))
    print(at2.mass2ion(100,'[M-2H]2-'))
    print(at2.adduct_list(mode = 'negative'))
    
