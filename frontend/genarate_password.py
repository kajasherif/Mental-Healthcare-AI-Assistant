import streamlit_authenticator as stauth

val = input("Enter your password to hash: ")
hashed_passwords = stauth.Hasher([val]).generate()
print(hashed_passwords)
