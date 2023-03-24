import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# countries chosen for analysis
chosenCountries = ['China', 'Canada', 'India', 'Pakistan',
                   'United Kingdom', 'Australia', 'United States', 'Nigeria']


def getDataframes(filename):
    '''
    Takes filename as argument.\n
    Returns two dataframes, one with years as columns and second with countries as columns.
    '''

    yearData = pd.read_csv(filename, skiprows=3, index_col='Country Name')

    # drop the unnecessary columns
    yearData.drop(['Country Code', 'Indicator Name',
                   'Indicator Code'], axis=1, inplace=True)

    # filter out the chosen countries
    yearData = yearData.loc[chosenCountries]

    # drop the columns (years) which have all null values
    yearData.dropna(axis=1, how='all', inplace=True)

    return yearData, yearData.transpose()


def getCommonYears(yearList1, yearList2):
    '''
    Takes two lists of years and returns a list of common years between them.
    '''
    return sorted(list(set(yearList1) & set(yearList2)))


def selectYears(yearsList, div=5):
    '''
    Takes a list of years and selects the years that are divisible by `div`.
    '''
    return [year for year in yearsList if not (int(year) % div)]


yearGDP, countryGDP = getDataframes('./Data/GDPPerCapita$.csv')
yearCO2, countryCO2 = getDataframes('./Data/CO2EmissionsTonPerCapita.csv')
yearMethane, countryMethane = getDataframes('./Data/MethaneEmissionTotal.csv')
yearPopu, countryPopu = getDataframes('./Data/PopulationTotal.csv')
yearPower, countryPower = getDataframes(
    './Data/PowerConsumptionkWhPerCapita.csv')
yearForest, countryForest = getDataframes('./Data/ForestArea%.csv')
yearArable, countryArable = getDataframes('./Data/ArableLand%.csv')
yearAgri, countryAgri = getDataframes('./Data/AgriculturalLand%.csv')


# GDP per CO2 Emission

selectedYears = selectYears(getCommonYears(yearCO2.columns, yearGDP.columns))
GDPPerCO2 = yearGDP[selectedYears] / yearCO2[selectedYears]

fig = plt.figure(figsize=(20, 20))
for country in chosenCountries:
    ax = plt.plot(GDPPerCO2.loc[country], label=country)

font = dict(size=25)
plt.legend(fontsize='xx-large')
plt.title('Energy Efficiency', fontdict=font)
plt.xticks(fontsize=20)
plt.xlabel('Year', fontdict=font)
plt.yticks(fontsize=20)
plt.ylabel('GDP ($) / CO2 Emission (Ton)', fontdict=font)
plt.savefig('EneryEfficiency')
