# In this problem you will write a program to "encrypt" and "decrypt" strings of text using a "password".
# The cryptogram will be based on letter rotation, with the password indicating how far each letter of the plain text should be rotated. The letter 'a' rotates by 0, 'b' by 1, 'c' by 2, and so forth. For example, 'z' rotated by 'b' (1) would become 'a' again, since in rotation, 'z' is followed by 'a' again.
# For example, if the password is "abc" and the plain text is "aaaaaa", then the encrypted text is "abcabc", with each letter of the password effecting one letter of plain text. The password is just repeated as long as necessary to encrypt all the text.
# Only letters should be encrypted, and they should maintain their case. So password "abc" would encrypt 'Hello World!' as 'Hfnlp Yosnd!'.
# Passwords are not case-sensitive, and characters in the password that are not letters are ignored/skipped/as if they were not there. A blank password leaves the plain text unchanged.
# Write two functions: encrypt and decrypt. Each function will take an input string and a password string, and will return the encrypted or decrypted string output.



def process_text(text, password, encrypt=True):
  
   filtered_password = [rot for rot in password.lower() if rot.isalpha()] # To ensure only alphabetical characters are there in password
   if not filtered_password:  # To ensure no empty password
     return text
     
   rotation_values = [ord(c) - ord('a') for c in filtered_password] # password --> rotation_value
   
   result = []
   rotation_index = 0
   for char in text:
          if char.isalpha():
            
              if encrypt:
                    direction = 1 # encrypt
              else: 
                    direction = -1 # decrypt
                
              # here I apply the desired rotation
              if char.isalpha():
                    if char.islower():
                      alpha_start = ord('a')
                    else:
                      alpha_start = ord('A')
                    temp = ( ord(char) - alpha_start + (direction * rotation_values[rotation_index]) ) % 26
                    rotated_char = chr(alpha_start + temp)
              else:
                    rotated_char = char
              result.append(rotated_char)
              rotation_index = (rotation_index + 1) % len(rotation_values)     # progress to next rotation_value
              
          else:
              result.append(char)
          
   return ''.join(result)
      
       
       


def encrypt (text, password):
    return process_text(text, password, encrypt=True)
def decrypt (text, password):
    return process_text(text, password, encrypt=False)
    
    
    
# Driver
password = "abc"
plain_text = "Hello World!"
encrypted_text = encrypt(text=plain_text, password=password)
decrypted_text = decrypt(text=encrypted_text, password=password)


print("Password:", password)
print("Plain text:", plain_text)
print("Encrypted text: ", encrypted_text)

if decrypted_text == plain_text:
    print(" ")
    print("----------Decryption successful-----------")
    print("Decrypted text: ", decrypted_text)

else:
    print(" ")
    print("----------- Decryption failed -----------")
    print("Decrypted text: ", decrypted_text)
    
    




### Tests conducted

# password: "abc"
# plain text: ""
# encrypted text: ""
# decrypted text: ""

# password: ""
# plain text: "Hello World!"
# encrypted text: "Hello World!"
# decrypted text: "Hello World!"

# password: "1234567890"
# plain text: "Hello World!"
# encrypted text: "Hello World!"
# decrypted text: "Hello World!"

# password: "abc"
# plain text: "1234567890!@#$%^&*()"
# encrypted text: "1234567890!@#$%^&*()"
# decrypted text: "1234567890!@#$%^&*()"

# password: "aBc"
# plain text: "Hello World!"
# encrypted text: "Hfnlp Yosnd!"
# decrypted text: "Hello World!"

# password: "abc"
# plain text: "The quick brown fox jumps over the lazy dog"
# encrypted text: "Tig qvkcl drpyn gqx kwmqu owgr uje mczz foh"
# decrypted text: "The quick brown fox jumps over the lazy dog"

# password: "abcdefg"
# plain text: "Hi"
# encrypted text: "Hj"
# decrypted text: "Hi"

# password: "ab1c!d@e#"
# plain text: "Testing123"
# encrypted text: "Tfuwmnh123"
# decrypted text: "Testing123"

# password: "abc"
# plain text: "Hello, World! 123."
# encrypted text: "Hfnlp, Yosnd! 123."
# decrypted text: "Hello, World! 123."
