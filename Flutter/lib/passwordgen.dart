import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:unhashed/main.dart';
import 'package:flutter/services.dart';

class SuggestedPasswordPage extends StatefulWidget {
  const SuggestedPasswordPage({super.key, required this.password});

  final String password;

  @override
  _SuggestedPasswordPageState createState() => _SuggestedPasswordPageState();
}

class _SuggestedPasswordPageState extends State<SuggestedPasswordPage> {
  var result = {};
  @override
  void initState() {
    super.initState();
    _getsuggestions();
  }

  Future<void> _getsuggestions() async {
    // URL for your Django password strength endpoint
    final url = 'http://10.0.2.2:8000/analysis/suggestions/';

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
        setState(() {
          // Refresh UI with new data
          result = json.decode(response.body);
        });
        print(result);
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
        content: SelectableText(message,
            style: TextStyle(
              color: Colors.white,
            )),
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
  Widget build(BuildContext context) {
    void copyToClipboard(String text, BuildContext context) {
      Clipboard.setData(ClipboardData(text: text));
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Copied to clipboard!')),
      );
    }

    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21),
      body: RefreshIndicator(
        onRefresh: _getsuggestions,
        child: Stack(children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: ListView(
              children:[ Column(
                  mainAxisAlignment: MainAxisAlignment.start,
                  children: [
                    Text(
                      'Your AI-Generated Secure Password',
                      style: GoogleFonts.kantumruyPro(
                        textStyle: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 175),
                    Text(
                      "Improved Password from original password",
                      style: GoogleFonts.kantumruyPro(
                        textStyle: const TextStyle(
                          fontSize: 20,
                          color: Colors.white,
                        ),
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 10),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                              vertical: 12, horizontal: 24),
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
                            (result['suggestions'] != null &&
                                    result['suggestions']
                                            ['Improved Original'] !=
                                        null)
                                ? result['suggestions']['Improved Original']
                                : "Loading...",
                            style: GoogleFonts.kantumruyPro(
                              textStyle: const TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ),
                        IconButton(
                          icon: Icon(Icons.copy),
                          onPressed: () {
                            copyToClipboard(
                                (result['suggestions'] != null &&
                                        result['suggestions']
                                                ['Improved Original'] !=
                                            null)
                                    ? result['suggestions']
                                        ['Improved Original']
                                    : "Loading...",
                                context);
                          },
                        ),
                      ],
                    ),
                    const SizedBox(height: 40),
                    Text(
                      "Secure Passphrase",
                      style: GoogleFonts.kantumruyPro(
                        textStyle: const TextStyle(
                          fontSize: 20,
                          color: Colors.white,
                        ),
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 10),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                              vertical: 12, horizontal: 24),
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
                            (result['suggestions'] != null &&
                                    result['suggestions']
                                            ['Secure Passphrase'] !=
                                        null)
                                ? result['suggestions']['Secure Passphrase']
                                : "Loading...",
                            style: GoogleFonts.kantumruyPro(
                              textStyle: const TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ),
                        IconButton(
                          icon: Icon(Icons.copy),
                          onPressed: () {
                            copyToClipboard(
                                (result['suggestions'] != null &&
                                        result['suggestions']
                                                ['Secure Passphrase'] !=
                                            null)
                                    ? result['suggestions']
                                        ['Secure Passphrase']
                                    : "Loading...",
                                context);
                          },
                        ),
                      ],
                    ),
                    const SizedBox(height: 40),
                    Text(
                      "Strong Random Password",
                      style: GoogleFonts.kantumruyPro(
                        textStyle: const TextStyle(
                          fontSize: 20,
                          color: Colors.white,
                        ),
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 10),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                              vertical: 12, horizontal: 24),
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
                            (result['suggestions'] != null &&
                                    result['suggestions']
                                            ['Random Password'] !=
                                        null)
                                ? result['suggestions']['Random Password']
                                : "Loading...",
                            style: GoogleFonts.kantumruyPro(
                              textStyle: const TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ),
                        IconButton(
                          icon: Icon(Icons.copy),
                          onPressed: () {
                            copyToClipboard(
                                (result['suggestions'] != null &&
                                        result['suggestions']
                                                ['Random Password'] !=
                                            null)
                                    ? result['suggestions']['Random Password']
                                    : "Loading...",
                                context);
                          },
                        ),
                      ],
                    ),
                  ]),
        ]),
          ),
          Positioned(
            bottom: 30,
            left: 16,
            right: 16,
            child: SizedBox(
              width: 250,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(context,
                      MaterialPageRoute(builder: (context) => MyHomePage()));
                },
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                      vertical: 14, horizontal: 24),
                  textStyle:
                      const TextStyle(fontSize: 18, color: Colors.white),
                  backgroundColor: const Color(0xFF0A0E21),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                    side:
                        const BorderSide(color: Colors.blueAccent, width: 2),
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(
                      Icons.home,
                      color: Colors.white,
                    ),
                    SizedBox(
                      width: 10,
                    ),
                    Text("Return Home",
                        style: GoogleFonts.kantumruyPro(
                            fontSize: 18, color: Colors.white))
                  ],
                ),
              ),
            ),
          ),
        ]),
      ),
    );
  }

  void copyToClipboard(String text, BuildContext context) {
    Clipboard.setData(ClipboardData(text: text));
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Copied to clipboard!')),
    );
  }
}
