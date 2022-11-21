import sys
import os


def disable_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def alike_match(strings, substrings):
    def anymatch(string):
        for substring in substrings:
            if substring in string:
                return True

        return False

    return [anymatch(string) for string in strings]


def alike_matches(strings, substrings):
    matches = []

    for string in strings:
        for i, substring in enumerate(substrings):
            if substring in string:
                #print(f"{substring} in {string}")
                matches.append(string)

    return matches
