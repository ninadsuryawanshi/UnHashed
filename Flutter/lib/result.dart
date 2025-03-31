

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:unhashed/loading.dart';
import 'package:unhashed/main.dart';
import 'package:unhashed/passwordgen.dart';

class PasswordAnalysisPage extends StatefulWidget {
  const PasswordAnalysisPage({super.key, required this.passwordData});
  final Map<String, dynamic> passwordData;

  @override
  State<PasswordAnalysisPage> createState() => _PasswordAnalysisPageState();
}

class _PasswordAnalysisPageState extends State<PasswordAnalysisPage> {
  
  Map<String, dynamic> getStrengthCategory(double score) {
    if (score < 30) {
      return {'category': 'Very Weak', 'emoji': '🔴', 'recommendation': 'Change immediately!'};
    } else if (score < 50) {
      return {'category': 'Weak', 'emoji': '🟠', 'recommendation': 'Needs significant improvement'};
    } else if (score < 65) {
      return {'category': 'Moderate', 'emoji': '🟡', 'recommendation': 'Could be stronger'};
    } else if (score < 80) {
      return {'category': 'Strong', 'emoji': '🟢', 'recommendation': 'Good password'};
    } else {
      return {'category': 'Very Strong', 'emoji': '🟢', 'recommendation': 'Excellent password'};
    }
  }


  @override
  Widget build(BuildContext context) {
    double finalScore=widget.passwordData['final_score'];
        var strength = getStrengthCategory(finalScore);
        String password = widget.passwordData['password'];

    return WillPopScope(
      onWillPop: () async {
        Navigator.push(context, MaterialPageRoute(
          builder: (context) => MyHomePage(),
        ));
        return true;
      },
      child: Scaffold(
        appBar: AppBar(title: const Text('Password Security Analysis')),
        body: RefreshIndicator(
          onRefresh: () {
            Navigator.push(context, MaterialPageRoute(
              builder: (context) => Loading(password, password: password, 
              ),
            ));
            return Future.value();  
          },
          child: Stack(
            children: [
              Padding(
                padding: const EdgeInsets.only(bottom: 130.0), // Leave space for button
                child: SingleChildScrollView(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                '${strength['emoji']} PASSWORD STRENGTH: ${strength['category']} (${finalScore.toStringAsFixed(1)}/100)',
                 style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),),
              ),
              Divider(),
              Text('📊 SUMMARY:', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),),),
              Text(' • Length: ${password.length} characters', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
              Text(' • Time to Crack: ${widget.passwordData['crack_result']['formatted_time']}', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
              Text(' • Found in Breaches: ${widget.passwordData['leak_result']['exact_match'] ? "Yes" : "No"}', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
              if (!widget.passwordData['leak_result']['exact_match']) Text(' ✓ NOT FOUND IN DATA BREACHES', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
              if (widget.passwordData['leak_result']['rnn_confidence'] < 0.3)
                Text(' ✓ USES RARE PASSWORD PATTERN', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),),),
              Divider(),
              Text('🔒 SECURITY ASSESSMENT:', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),),),
              for (var reason in widget.passwordData['leak_result']['reasoning'] + [widget.passwordData['pattern_result']['reasoning']])
                if (reason.isNotEmpty && reason != "No significant weaknesses detected.")
                  Text(' • $reason', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
              Divider(),
              Text('📈 DETAILED METRICS:', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),),),
              Text(' • Leak analysis score: ${(1 - widget.passwordData['leak_result']['risk_score']) * 100}/100', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
              Text(' • Pattern analysis score: ${widget.passwordData['pattern_result']['strength_score']}/100', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
              Text(' • Entropy: ${widget.passwordData['crack_result']['entropy']} bits (${widget.passwordData['crack_result']['strength_rating']})', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
              Divider(),
              Text('💡 RECOMMENDATION:', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),),),
              Text(' ${strength['recommendation']}', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
                    const SizedBox(height: 20),
              if (finalScore < 60) ...[
                Text('Tips to improve your password:', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),),),
                if (password.length < 12) Text('  • Make it longer (at least 12 characters)', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
                if (!widget.passwordData['crack_result']['charset_details']['uppercase']) Text('  • Add uppercase letters', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
                if (!widget.passwordData['crack_result']['charset_details']['lowercase']) Text('  • Add lowercase letters', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
                if (!widget.passwordData['crack_result']['charset_details']['digits']) Text('  • Add numbers', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
                if (!widget.passwordData['crack_result']['charset_details']['symbols']) Text('  • Add special characters', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                     
                      color: Colors.white,
                    ),),),
                if (widget.passwordData['leak_result']['exact_match'] || widget.passwordData['leak_result']['similarity_score'] > 0.8) 
                  Text('  • Avoid common words and patterns', style: GoogleFonts.kantumruyPro(
                    textStyle: const TextStyle(
                      fontSize: 20,
                      
                      color: Colors.white,
                    ),),),
                      ],]
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
                      builder: (context) => SuggestedPasswordPage(password: widget.passwordData['password'],),
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
      ),
    );
  }

  
}