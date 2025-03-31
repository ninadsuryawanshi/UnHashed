import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:unhashed/result.dart';
import 'package:http/http.dart' as http;

class Loading extends StatefulWidget {
  const Loading(String s, {super.key, required this.password});
  final String password;
  @override
  State<Loading> createState() => _LoadingState();
}

class _LoadingState extends State<Loading> {
  final PageController _pageController = PageController();
   final List<String> texts = [
    "Hackers can crack a 6-character password in just 10 minutes",
    "Weak passwords are responsible for approximately 30% of global data breaches",
    "Over 60% of individuals admit to reusing passwords across multiple accounts",
    "An alarming 81% of company data breaches are caused by poor password practices",
    "In 2022, over 24 billion passwords were exposed by hackers"
  ];
  int _currentPage = 0;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _startAutoScroll();
    _checkPasswordStrength();
    print("Password: ${widget.password}");
  }

  void _startAutoScroll() {
    _timer = Timer.periodic(Duration(seconds: 5), (timer) {
      if (_currentPage < texts.length - 1) {
        _currentPage++;
      } else {
        _currentPage = 0; // Loop back to the first text
      }
      _pageController.animateToPage(
        _currentPage,
        duration: Duration(milliseconds: 500),
        curve: Curves.easeInOut,
      );
    });
  }

  Future<void> _checkPasswordStrength() async {
    // URL for your Django password strength endpoint
    final url = 'http://10.0.2.2:8000/analysis/analyze-password/';
    
    try {
      final Map<String, dynamic> requestBody = {"password": widget.password};
    print("Sending request body: ${jsonEncode(requestBody)}");
      // Send the password to Django
      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(requestBody),
      );
      
      if (response.statusCode == 200) {
        // Got password strength result
        final result = json.decode(response.body);
        
        // Navigate to result screen with the strength info
        Navigator.push(
          // ignore: use_build_context_synchronously
          context,
          MaterialPageRoute(
            builder: (context) => PasswordAnalysisPage(
              passwordData: result,

            ),
          ),
        );
      } else {
        // Something went wrong
        _showErrorDialog('Error checking password: ${response.statusCode}');
      }
    } catch (error) {
      // Network or other error
      _showErrorDialog('Network error: $error');
    }
  }
  
  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Error'),
        content: SelectableText(message, style: TextStyle(color: Colors.white,)),
        actions: <Widget>[
          TextButton(
            child: Text('Okay'),
            onPressed: () {
              Navigator.of(ctx).pop();
            },
          )
        ],
      ),
    );
  }

  @override
  void dispose() {
    _timer?.cancel(); // Cancel timer when widget is disposed
    _pageController.dispose();
    super.dispose();
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              SizedBox(height: 150),
              Container(
            height: 100,
            margin: const EdgeInsets.symmetric(horizontal: 35),
            child: PageView.builder(
              controller: _pageController,
              itemCount: texts.length,
              physics: NeverScrollableScrollPhysics(),
              itemBuilder: (context, index) {
                return Center(
                  child: Text(
                    texts[index],
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white),
                  ),
                );
              },
            ),
          ),SizedBox(height: 150),
              const CircularProgressIndicator(
                color: Colors.blue,
              ),
              const SizedBox(height: 50),
              Text(
                "Analyzing Password Strength...",
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 50),
              
            ],
          ),
        ),
      ),
    );
  }
}