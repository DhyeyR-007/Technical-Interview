// Given a txt file (Kodiak_data.txt) and number_string write a code that will output alphabetically ordered possible and valid words.
// The Kodiak_data.txt file has a list of all possible and valid words needed


#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_set>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace std;

// Feel free to modify any of this starter code however you want. 

static const unordered_map<int, vector<char>> kT9Mapping{{2, {'a', 'b', 'c'}},
                                                         {3, {'d', 'e', 'f'}},
                                                         {4, {'g', 'h', 'i'}},
                                                         {5, {'j', 'k', 'l'}},
                                                         {6, {'m', 'n', 'o'}},
                                                         {7, {'p', 'q', 'r', 's'}},
                                                         {8, {'t', 'u', 'v'}},
                                                         {9, {'w', 'x', 'y', 'z'}}};

vector<string> ReadWords() {
  vector<string> valid_words;
  ifstream word_file("/home/coderpad/data/words.txt");
  if (word_file.is_open()) {
    string word;
    while (getline(word_file, word)) {
      valid_words.push_back(word);
    }
    word_file.close();
  }
  return valid_words;
}


// ##################################################################

vector<string> GetPossibleWords(const string& number_string) {
  vector<string> possible_words;
  // Please write your solution here.

  
  // Check if the number string contains any invalid digits
  for (char digit : number_string) {
    if (digit < '2' || digit > '9') {
      return possible_words;  // Return an empty vector if invalid digits are found
    }
  }
  

  const auto word_list = ReadWords();
  for (const string& word : word_list) {
    
    if (word.size() != number_string.size()) {
      continue;
    }

    bool valid = true;
    for (size_t i = 0; i < word.size(); i++) {
      char character = word[i];
      int digit = number_string[i] - '0';
      const auto& characters = kT9Mapping.at(digit);
      if (find(characters.begin(), characters.end(), character) == characters.end()) {
        valid = false;
        break;
      }
    }

    if (valid) {
      possible_words.push_back(word);
    }
  }


  sort(possible_words.begin(), possible_words.end());


  cout << "Vector of possible, valid words produced for the string of digits " << number_string << ":" << endl;
  for_each(possible_words.begin(),
             possible_words.end(),
             [](const auto& elem) {
                 cout << elem << " " ;
             });
  cout << endl << endl;


  //// possible_words is a vector of possible, valid words in alphabetical order for the given string of digits
  return possible_words;
}



// ##################################################################

// bool PossibleWordsMatch(const vector<string>& expected, const vector<string>& computed) {
//   return expected == computed;
// }

bool PossibleWordsMatch(const vector<string>& expected, const vector<string>& computed) {
  

    //// size comparison
    if (expected.size() != computed.size()) {
      return false;
    }

    //// Create copies of expected and computed vectors for case-insensitive comparison
    vector<string> expectedLower = expected;
    vector<string> computedLower = computed;

    //// Convert strings to lowercase to make the code case insensitive
    for (auto& str : expectedLower) {
      transform(str.begin(), str.end(), str.begin(), ::tolower);
    }
    for (auto& str : computedLower) {
      transform(str.begin(), str.end(), str.begin(), ::tolower);
    }

    //// Sort the expected lowercase vectors since I need this expected inputs to match with the alphabetically arranged possible_words vector
    sort(expectedLower.begin(), expectedLower.end());

    //// Compare the sorted lowercase vectors
    return expectedLower == computedLower;
}



/* Please briefly describe the run time of your solution below:


NOTE: (i) The below mentioned time complexity analysis is considering the worst case scenario i.e. the upper limit of time complexity.
      (ii) SINCE MY SOLUTION ALSO INVOLVES CHANGING "PossibleWordsMatch" FUNCTION HENCE I ALSO EXPLAIN ITS TIME COMPLEXITY;



GetPossibleWords : Time complexity of the "GetPossibleWords function", is explained below w.r.t. all the steps:

    (1) Checking for invalid digits (Number string for loop) O(M)
        The loop iterates over the characters in the number_string, where M is the length of the number_string. It has a linear time complexity of O(M).

    (2) ReadWords() function: O(N * K)
        The ReadWords function reads the words from a file and returns a vector of words. This operation depends on the number of words in the file, denoted as N, and the average length of each word or worst case if each word has maximum/same length, denoted as K. So, reading the words has a time complexity of O(N * K).

    (3) 'word.size() != number_string.size()': O(1) constant time complexity.

    (4) Character comparison: O(N * K * 4)
        The nested loop iterates over each word in the word_list and performs character comparisons. The outer loop iterates N times, and the inner loop iterates K times for each word. Within the inner loop, the find function performs a linear search in a vector of characters of size 4 (the maximum number of characters for a digit). Therefore, the time complexity is O(N * K * 4).

    (5) Adding the word to the possible_words vector: O(N)
        After checking the validity of each word, the valid words are stored in the possible_words vector. The number of valid words stored is proportional to the number of words in the word_list, which is N. So, adding the words to the possible_words vector has a time complexity of O(N).

    (6) Sorting the possible_words vector: O(N log N)
        The sort function is applied to the possible_words vector for sorting in alphabetical order, which contains N words. Sorting N elements has an average time complexity of O(N log N).

    (7) Printing the vector: O(N)
        The code snippet prints the elements of the possible_words vector in alphabetical order, which requires iterating over each word and printing it. The time complexity is proportional to the maximum possible number of elements in the vector, which is N.

    Therefore, overall time complexity of "GetPossibleWords function", O(M) + O(N * K) + O(N * K * 4) + O(N) + O(N log N) + O(N) 

    Now after simplifying we get:
        O(M) + O(N * K) + O(N) + O(N log N)

    Now if our goal is to output possible valid words (no empty output) everytime, then under this implicit assumption we consider M < N, hence:
        O(N * K) + O(N) + O(N log N)

    Further simplifying this we get;
        O(N * K) + O(N log N)

    Therefore, the final time complexity without any further assumptions, can be considered here as : O(N * (K + log n))
    



---------------------------------------------------------------------------------------------------------------------------




PossibleWordsMatch : Time complexity of the "PossibleWordsMatch function",, is explained below w.r.t. all the steps:

    (1) Size Comparison: The function compares the sizes of the expected and computed vectors using expected.size() and computed.size(). This operation has a constant time complexity of O(1).

    (2) Copying and Lowercasing: The function creates copies of the expected and computed vectors, and then converts all strings to lowercase using transform and ::tolower. The time complexity of this operation is O(N), where N is the total number of characters in both vectors. Since we perform this operation separately for both vectors, the overall time complexity remains O(N).

    (3) Sorting: The lowercase copy of the vector, expectedLower is sorted using sort. The time complexity of the sorting operation is O(N log N), where N is the total number of strings in the vectors.

    (4) Comparison: The sorted lowercase vectors expectedLower and computedLower are compared using the == operator. The comparison involves comparing each element of the vectors, which has a time complexity of O(N). Thus, the overall time complexity of the comparison operation is O(N).

    Therefore, the overall time complexity of "PossibleWordsMatch function", O(1) + O(N) + O(N log N) + O(N) simplifies to O(N log N).



 */


TEST_CASE("Possible words match", "[GetPossibleWords]") {
  const auto word_list = ReadWords();
  // Build your data structure to hold the valid words here and then pass it to GetPossibleWords.
  REQUIRE(PossibleWordsMatch({"act", "bat", "cat"}, GetPossibleWords("228")));
  REQUIRE(PossibleWordsMatch({"kodiak"}, GetPossibleWords("563425")));

  // Add your test cases here.
  //// My test case 1
  REQUIRE(PossibleWordsMatch({}, GetPossibleWords("")));
  //// My test case 2
  REQUIRE(PossibleWordsMatch({}, GetPossibleWords("111")));
  //// My test case 3
  REQUIRE(PossibleWordsMatch({"kohls"}, GetPossibleWords("56457")));
  //// My test case 4
  REQUIRE(PossibleWordsMatch({"FED","EfF", "dee", "fee"}, GetPossibleWords("333")));
  //// My test case 5
  REQUIRE(PossibleWordsMatch({}, GetPossibleWords("23456789999987655754654554465456688")));
  //// My test case 6
  REQUIRE(PossibleWordsMatch({"id","if","HE"}, GetPossibleWords("43")));

}
