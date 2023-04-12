import csv
import statistics

start = 5

rating = []
total_gestures = []
missed_gestures = []
incorrect_gestures = []
correct_gestures = []

n = 0

# read in data
with open('data.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)

    curr_row = 0

    for row in reader:
        curr_row += 1
        if (curr_row <= start):
            continue
        n += 1
        rating.append(int(row[2]))
        total_gestures.append(int(row[5]))
        correct_gestures.append(int(row[6]))
        missed_gestures.append(int(row[7]))
        incorrect_gestures.append(int(row[8]))

# calculate percents
for i in range(0,len(rating)):
    missed_gestures[i] = missed_gestures[i] / total_gestures[i]
    correct_gestures[i] = correct_gestures[i] / total_gestures[i]
    incorrect_gestures[i] = incorrect_gestures[i] / total_gestures[i]

# calculate means
missed_mean = sum(missed_gestures) / len(missed_gestures)
incorrect_mean = sum(incorrect_gestures) / len(incorrect_gestures)
correct_mean = sum(correct_gestures) / len(correct_gestures)
rating_mean = sum(rating) / len(rating)

# calculate std deviation
missed_stdev = statistics.stdev(missed_gestures)
incorrect_stdev = statistics.stdev(incorrect_gestures)
correct_stdev = statistics.stdev(correct_gestures)
rating_stdev = statistics.stdev(rating)

print("RATING\nMean:%f\nStandard Deviation:%f\n" % (rating_mean, rating_stdev))
print("CORRECT\nMean:%f\nStandard Deviation:%f\n" % (correct_mean, correct_stdev))
print("MISSED\nMean:%f\nStandard Deviation:%f\n" % (missed_mean, missed_stdev))
print("INCORRECT\nMean:%f\nStandard Deviation:%f\n" % (incorrect_mean, incorrect_stdev))





