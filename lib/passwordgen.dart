import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:unhashed/main.dart';
import 'package:flutter/services.dart';

class SuggestedPasswordPage extends StatelessWidget {
  const SuggestedPasswordPage({super.key, required this.suggestedPassword});

  final String suggestedPassword;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Text(
                'Your AI-Generated Secure Password',
                style: GoogleFonts.kantumruyPro(
                  textStyle: const TextStyle(
                    fontSize: 20,
                    
                    color: Colors.white,
                  ),
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              Container(
                padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 24),
                decoration: BoxDecoration(
                  
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: [
                    BoxShadow(
                      color: const Color.fromARGB(255, 34, 96, 205),
                      blurRadius: 10,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Text(
                  suggestedPassword,
                  style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              const SizedBox(height: 30),
              SizedBox(
                width: 250,
                child: ElevatedButton(
                  onPressed: () {
                    Clipboard.setData(ClipboardData(text: suggestedPassword));
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text("Password copied to clipboard!")),
                    );
                  },
                  
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 24),
                    textStyle: const TextStyle(fontSize: 18, color: Colors.white),
                    backgroundColor: Colors.blueAccent,
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [const Icon(Icons.copy, color: Colors.white,),SizedBox(width: 10,), Text("Copy Password", style: GoogleFonts.kantumruyPro(fontSize: 18, color: Colors.white))],
                  ),
                ),
              ),
              const SizedBox(height: 70),
              SizedBox(
                width: 250,
                child: ElevatedButton(
                  onPressed: () {
                    
                    Navigator.push(context, MaterialPageRoute(
                      builder: (context) => MyHomePage()
                    ));
                  },
                  
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 24),
                    textStyle: const TextStyle(fontSize: 18, color: Colors.white),
                    backgroundColor: const Color(0xFF0A0E21),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                      side: const BorderSide(color: Colors.blueAccent, width: 2),
                    ),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [const Icon(Icons.home, color: Colors.white,),SizedBox(width: 10,), Text("Return Home", style: GoogleFonts.kantumruyPro(fontSize: 18, color: Colors.white))],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
      backgroundColor: const Color(0xFF0A0E21), // Dark Navy Blue
    );
  }
}
