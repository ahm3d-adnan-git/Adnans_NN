#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <iostream>
#include <string>
#include <fstream>

unsigned int swapEndian(unsigned int num) {
    return ((num >> 24) & 0xFF) |      // Move byte 3 to byte 0
           ((num << 8) & 0xFF0000) |  // Move byte 1 to byte 2
           ((num >> 8) & 0xFF00) |    // Move byte 2 to byte 1
           ((num << 24) & 0xFF000000);// Move byte 0 to byte 3
}

bool compareWithExpectedFile(const char*& generatedFileName,const char*& expectedFileName) {
    std::ifstream expectedOutputFile(expectedFileName, std::ios::binary);
    std::ifstream generatedOutputFile(generatedFileName,std::ios::binary);
    if (!expectedOutputFile) {
        std::cerr << "Error opening file: " << expectedFileName << std::endl;
        return false;
    }
    if (!generatedOutputFile) {
        std::cerr << "Error opening file: " << generatedFileName << std::endl;
        return false;
    }


    std::string genLine, expectedLine;

    size_t lineNumber = 0;
    bool allMatch = true;
    while (std::getline(generatedOutputFile, genLine) && std::getline(expectedOutputFile, expectedLine)) {
        lineNumber++;
        if (genLine != expectedLine) {
            std::cerr << "Mismatch at line " << lineNumber << ":\n";
            std::cerr << "Generated: " << genLine << "\n";
            std::cerr << "Reference: " << expectedLine << "\n";
            allMatch = false;
            break;
        }
    }
    expectedOutputFile.close();
    generatedOutputFile.close();
    return allMatch;

}

#endif // UTILITIES_HPP