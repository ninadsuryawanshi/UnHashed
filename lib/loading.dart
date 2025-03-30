import 'dart:async';

import 'package:flutter/material.dart';
import 'package:unhashed/result.dart';

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
              ElevatedButton(onPressed: (){
                Navigator.push(context, MaterialPageRoute(
                  builder: (context) => PasswordAnalysisPage(password: widget.password),
                ));
              }, child: Text("Go Ahead (Temporary)"), ),
            ],
          ),
        ),
      ),
    );
  }
}