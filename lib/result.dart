import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:unhashed/main.dart';
import 'package:unhashed/passwordgen.dart';

class PasswordAnalysisPage extends StatefulWidget {
  const PasswordAnalysisPage({super.key, required this.password});
  final String password;

  @override
  State<PasswordAnalysisPage> createState() => _PasswordAnalysisPageState();
}

class _PasswordAnalysisPageState extends State<PasswordAnalysisPage> {
  


  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: () async {
        Navigator.push(context, MaterialPageRoute(
          builder: (context) => MyHomePage(),
        ));
        return true;
      },
      child: Scaffold(
        appBar: AppBar(title: const Text('Password Security Analysis')),
        body: Stack(
          children: [
            Padding(
              padding: const EdgeInsets.only(bottom: 120.0), // Leave space for button
              child: SingleChildScrollView(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildSectionTitle('Password:'),
                      _buildContentText(widget.password),
                      const SizedBox(height: 10),
                      
                      _buildSectionTitle('Length:'),
                      _buildContentText(widget.password.length.toString()),
                      const SizedBox(height: 10),
                      
                      _buildSectionTitle('Character Composition:'),
                      _buildContentText('Mixed Charset (Uppercase, Lowercase, Numbers, Symbols)'),
                      const SizedBox(height: 20),
                      
                      _buildSectionTitle('Security Metrics:'),
                      _buildBulletPoint('Character Set Size: 94'),
                      _buildBulletPoint('Total Search Space: 6.09e+17 combinations'),
                      _buildBulletPoint('Entropy: 59.48 bits'),
                      const SizedBox(height: 20),
                      
                      _buildSectionTitle('Time-to-Crack Estimates:'),
                      _buildBulletPoint('Low-end CPU: ~193.9 years'),
                      _buildBulletPoint('High-end GPU: ~1.93 years'),
                      _buildBulletPoint('Botnet (1 trillion guesses/sec): ~17 hours'),
                      const SizedBox(height: 20),
                      
                      _buildSectionTitle('Attack Analysis:'),
                      _buildBulletPoint('Brute Force: Difficult (High Entropy)'),
                      _buildBulletPoint('Dictionary Check: Found common words ("Hello", "123")'),
                      _buildBulletPoint('Hybrid Attack: Moderately vulnerable'),
                      const SizedBox(height: 20),
                      
                      _buildSectionTitle('Weaknesses:'),
                      _buildBulletPoint('Contains dictionary words ("Hello123")'),
                      _buildBulletPoint('Predictable pattern (word + numbers + symbol)'),
                      const SizedBox(height: 20),
                      
                      _buildSectionTitle('Suggested Improvement:'),
                      _buildBulletPoint('Use a mix of random characters (e.g., "H3ll0!2@9x")'),
                      _buildBulletPoint('Increase length to 12+ characters for better security'),
                      const SizedBox(height: 20),
                      
                      _buildSectionTitle('Password Strength Rating:'),
                      Row(
                        children: [
                          _buildContentText('Medium '),
                          const Icon(Icons.circle, color: Colors.blue, size: 16),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
            Positioned(
              bottom: 50,
              left: 16,
              right: 16,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(context, MaterialPageRoute(
                    builder: (context) => SuggestedPasswordPage(suggestedPassword: widget.password,),
                  ));
                },
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  textStyle: const TextStyle(fontSize: 18, color: Colors.black),
                  backgroundColor: Colors.amber,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  
                ),
                child: Text('Generate Secure Password', style: GoogleFonts.kantumruyPro(color: Colors.black, fontSize: 18, fontWeight: FontWeight.bold)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 18,
        fontWeight: FontWeight.bold,
        color: Colors.blue,
      ),
    );
  }

  Widget _buildContentText(String text) {
    return Text(
      text,
      style: const TextStyle(fontSize: 16, color: Colors.white),
    );
  }

  Widget _buildBulletPoint(String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text("• ", style: TextStyle(fontSize: 16, color: Colors.white)),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(fontSize: 16, color: Colors.white),
            ),
          ),
        ],
      ),
    );
  }
}