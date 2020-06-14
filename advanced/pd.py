
import pandas as pd

def advancedStats(data, labels):
    '''Advanced stats should leverage pandas to calculate
    some relevant statistics on the data.

    data: numpy array of data
    labels: numpy array of labels
    '''
    # convert to dataframe
    df = pd.DataFrame(data)

    # print skew and kurtosis for every column
    for i in range(len(df.columns)):
        print("\nColumn {} stats:".format(i))
        column = df[df.columns[i]]
        skew = column.skew()
        kurt = column.kurtosis()
        print("Skewness: {} \t Kurtosis: {}".format(skew, kurt))
        
    # assign in labels
    df['labels'] = labels

    print("\n\nDataframe statistics")

    # groupby labels into "benign" and "malignant"
    group = df.groupby('labels')

    # collect means and standard deviations for columns,
    # grouped by label
    meansB = group.get_group('B').mean()
    sdevsB = group.get_group('B').std()
    meansM = group.get_group('M').mean()
    sdevsM = group.get_group('M').std()

    # Print mean and stddev for Benign
    print("Benign Stats:")
    print("\nMean:")
    print(meansB)
    print("\nStd:")
    print(sdevsB)

    print("\n")

    # Print mean and stddev for Malignant
    print("Malignant Stats:")
    print("\nMean:")
    print(meansM)
    print("\nStd:")
    print(sdevsM)