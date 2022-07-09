def encodeLZ(FileIn, FileOut):
    input_file = open(FileIn, 'r')
    encoded_file = open(FileOut, 'w')
    text_from_file = input_file.read()
    dict_of_codes = {text_from_file[0]: '1'}
    encoded_file.write('0' + text_from_file[0])
    text_from_file = text_from_file[1:]
    combination = ''
    code = 2
    for char in text_from_file:
        combination += char
        if combination not in dict_of_codes:
            dict_of_codes[combination] = str(code)
            if len(combination) == 1:
                encoded_file.write('0' + combination)
            else:
                encoded_file.write(dict_of_codes[combination[0:-1]] + combination[-1])
            code += 1
            combination = ''
    input_file.close()
    encoded_file.close()
    return True


def decodeLZ(FileIn, FileOut):
    coded_file = open(FileIn, 'r')
    decoded_file = open(FileOut, 'w')
    text_from_file = coded_file.read()
    dict_of_codes = {'0': '', '1': text_from_file[1]}
    decoded_file.write(dict_of_codes['1'])
    text_from_file = text_from_file[2:]
    combination = ''
    code = 2
    for char in text_from_file:
        if char in '1234567890':
            combination += char
        else:
            dict_of_codes[str(code)] = dict_of_codes[combination] + char
            decoded_file.write(dict_of_codes[combination] + char)
            combination = ''
            code += 1
    coded_file.close()
    decoded_file.close()


encodeLZ('input.txt', 'encoded.txt')
decodeLZ('encoded.txt', 'decoded.txt')
